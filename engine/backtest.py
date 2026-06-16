"""
Backtest Engine
===============
Simulates portfolio performance over a historical period.

Two strategies compared:
  1. Base (Buy & Hold)       - holds the portfolio as submitted, no rebalancing
  2. Regime-Adjusted         - rebalances when the detected regime changes,
                               applying each regime's allocation multipliers

Output:
  - Daily equity curves (cumulative return series)
  - Full performance metrics: return, volatility, Sharpe, Sortino,
    Max Drawdown, Calmar, win rate, best/worst month
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from engine.market_data import fetch_portfolio_prices
from engine.historical_regimes import reconstruct_regime_history


# Regime-based weight multipliers applied to each asset class
# These match the exposure map logic in the main engine
REGIME_MULTIPLIERS = {
    "SURGE":    {"equity": 1.20, "duration": 0.60, "credit": 1.20, "commodity": 1.10, "cash": 0.50},
    "CRUISE":   {"equity": 1.10, "duration": 0.80, "credit": 1.05, "commodity": 1.00, "cash": 0.70},
    "PRESSURE": {"equity": 0.85, "duration": 0.70, "credit": 0.90, "commodity": 1.30, "cash": 1.10},
    "SLIDE":    {"equity": 0.65, "duration": 1.20, "credit": 0.70, "commodity": 0.80, "cash": 1.40},
    "SHOCK":    {"equity": 0.35, "duration": 1.40, "credit": 0.50, "commodity": 0.60, "cash": 2.00},
    "FOG":      {"equity": 0.90, "duration": 0.95, "credit": 0.90, "commodity": 0.95, "cash": 1.10},
}

# Simple equity/bond/cash classification for each ticker
TICKER_ASSET_CLASS = {
    # Equity
    "SPY": "equity", "QQQ": "equity", "IWM": "equity", "VTI": "equity",
    "VXUS": "equity", "EEM": "equity", "VEA": "equity",
    "AAPL": "equity", "MSFT": "equity", "GOOGL": "equity", "AMZN": "equity",
    "NVDA": "equity", "META": "equity", "TSLA": "equity", "BRK-B": "equity",
    # Duration
    "TLT": "duration", "IEF": "duration", "BND": "duration", "AGG": "duration",
    "TIP": "duration", "VGLT": "duration", "GOVT": "duration",
    # Credit
    "HYG": "credit", "JNK": "credit", "LQD": "credit", "VCIT": "credit",
    # Commodities
    "GLD": "commodity", "IAU": "commodity", "SLV": "commodity",
    "DJP": "commodity", "GSG": "commodity", "USO": "commodity",
    # Cash / Short-term
    "SHV": "cash", "BIL": "cash", "SGOV": "cash",
    "VNQ": "equity",   # REITs treated as equity
    "VXUS": "equity",
}


def _get_asset_class(ticker: str) -> str:
    return TICKER_ASSET_CLASS.get(ticker.upper(), "equity")


def _apply_regime_weights(
    base_weights: dict,
    regime: str,
    profile_multiplier: float = 1.0,
) -> dict:
    """
    Apply regime multipliers to base portfolio weights.
    profile_multiplier: 0.55 conservative, 0.80 moderate, 1.0 aggressive
    """
    multipliers = REGIME_MULTIPLIERS.get(regime, REGIME_MULTIPLIERS["FOG"])
    adjusted = {}
    total = 0

    for ticker, weight in base_weights.items():
        asset_class = _get_asset_class(ticker)
        m = multipliers.get(asset_class, 1.0)
        # Blend: between full adjustment and neutral based on profile
        blended_m = 1.0 + (m - 1.0) * profile_multiplier
        adjusted[ticker] = weight * blended_m
        total += adjusted[ticker]

    # Renormalise to sum to 1
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


@dataclass
class BacktestMetrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    best_month: float
    worst_month: float
    max_drawdown_duration_days: int
    recovery_days: int
    cumulative_returns: list   # daily cumulative return values


def _compute_metrics(equity_curve: pd.Series, rf: float = 0.04) -> BacktestMetrics:
    """Compute full performance metrics from a daily equity curve."""
    daily_rets = equity_curve.pct_change().dropna()
    n_days = len(daily_rets)
    n_years = n_days / 252

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = daily_rets.std() * np.sqrt(252)

    # Sharpe
    daily_rf = rf / 252
    excess = daily_rets - daily_rf
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < daily_rf]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = (ann_return - rf) / downside_std if downside_std > 0 else 0

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Max drawdown duration
    in_drawdown = drawdown < -0.001
    dd_start = None
    max_dd_duration = 0
    recovery = 0
    for i, (date, is_dd) in enumerate(in_drawdown.items()):
        if is_dd and dd_start is None:
            dd_start = i
        elif not is_dd and dd_start is not None:
            duration = i - dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
                recovery = duration // 2
            dd_start = None

    # Win rate (monthly)
    monthly_rets = equity_curve.resample("ME").last().pct_change().dropna()
    win_rate = (monthly_rets > 0).mean() if len(monthly_rets) > 0 else 0.5
    best_month = monthly_rets.max() if len(monthly_rets) > 0 else 0
    worst_month = monthly_rets.min() if len(monthly_rets) > 0 else 0

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Cumulative return series (normalised to 0 start)
    cum_rets = (equity_curve / equity_curve.iloc[0] - 1).tolist()
    # Downsample to max 252 points for frontend
    if len(cum_rets) > 252:
        step = len(cum_rets) // 252
        cum_rets = cum_rets[::step]

    return BacktestMetrics(
        total_return=round(total_return, 4),
        annualized_return=round(ann_return, 4),
        annualized_vol=round(ann_vol, 4),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        calmar=round(calmar, 3),
        win_rate=round(win_rate, 3),
        best_month=round(best_month, 4),
        worst_month=round(worst_month, 4),
        max_drawdown_duration_days=max_dd_duration,
        recovery_days=recovery,
        cumulative_returns=cum_rets,
    )


def run_backtest(
    holdings: list,
    period_years: int = 3,
    rebalance_frequency: str = "monthly",
    profile_multiplier: float = 1.0,
) -> dict:
    """
    Run a real backtest for a given portfolio.

    holdings: list of {"ticker": str, "weight": float}
    period_years: lookback in years (1, 2, 3, or 5)
    rebalance_frequency: "monthly" or "on_regime_change"
    profile_multiplier: 0.55 conservative, 0.80 moderate, 1.0 aggressive

    Returns dict with "base" and "regime_adjusted" metrics.
    """
    from datetime import datetime, timedelta
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=period_years * 365 + 30)).strftime("%Y-%m-%d")

    # Normalise weights
    total_w = sum(h.get("weight", 0) for h in holdings)
    if total_w <= 0:
        # Equal weight fallback
        for h in holdings:
            h["weight"] = 100.0 / len(holdings)
        total_w = 100.0

    base_weights = {
        h["ticker"]: (h.get("weight", 0) / total_w)
        for h in holdings
        if h.get("ticker")
    }

    if not base_weights:
        raise ValueError("No valid holdings provided")

    # Fetch prices and regime history in parallel
    prices = fetch_portfolio_prices(list(base_weights.keys()), start=start_date, end=end_date)
    regime_series = reconstruct_regime_history(start=start_date, end=end_date)

    # Align dates
    common_dates = prices.index.intersection(regime_series.index)
    prices = prices.loc[common_dates]
    regime_series = regime_series.loc[common_dates]

    if len(prices) < 20:
        raise ValueError("Insufficient price history — try a shorter period or different tickers")

    # Daily returns for each holding
    daily_returns = prices.pct_change().fillna(0)

    # ── Base strategy (buy & hold) ──────────────────────────────────────────
    base_portfolio_returns = pd.Series(0.0, index=daily_returns.index)
    for ticker, weight in base_weights.items():
        if ticker in daily_returns.columns:
            base_portfolio_returns += daily_returns[ticker] * weight

    base_equity = (1 + base_portfolio_returns).cumprod()

    # ── Regime-adjusted strategy ────────────────────────────────────────────
    regime_portfolio_returns = pd.Series(0.0, index=daily_returns.index)
    current_weights = base_weights.copy()
    last_rebalance_regime = None

    for i, date in enumerate(daily_returns.index):
        current_regime = regime_series.loc[date]

        should_rebalance = False
        if rebalance_frequency == "on_regime_change":
            if current_regime != last_rebalance_regime:
                should_rebalance = True
                last_rebalance_regime = current_regime
        else:
            # Monthly: rebalance on first trading day of each month
            if i == 0 or date.month != daily_returns.index[i - 1].month:
                should_rebalance = True

        if should_rebalance:
            current_weights = _apply_regime_weights(
                base_weights, current_regime, profile_multiplier
            )

        day_return = 0.0
        for ticker, weight in current_weights.items():
            if ticker in daily_returns.columns:
                day_return += daily_returns[ticker].iloc[i] * weight

        regime_portfolio_returns.iloc[i] = day_return

    regime_equity = (1 + regime_portfolio_returns).cumprod()

    # ── Compute metrics ─────────────────────────────────────────────────────
    base_metrics = _compute_metrics(base_equity)
    regime_metrics = _compute_metrics(regime_equity)

    # Regime breakdown
    regime_counts = regime_series.value_counts()
    regime_breakdown = {
        r: round(regime_counts.get(r, 0) / len(regime_series), 3)
        for r in ["SURGE", "CRUISE", "PRESSURE", "SLIDE", "SHOCK", "FOG"]
    }

    return {
        "base": {
            "label": "Buy & Hold",
            "metrics": base_metrics.__dict__,
        },
        "regime_adjusted": {
            "label": f"Regime-Adjusted ({rebalance_frequency.replace('_', ' ').title()})",
            "metrics": regime_metrics.__dict__,
        },
        "regime_breakdown": regime_breakdown,
        "period_years": period_years,
        "start_date": str(prices.index[0].date()),
        "end_date": str(prices.index[-1].date()),
        "trading_days": len(prices),
        "tickers_used": list(base_weights.keys()),
    }
