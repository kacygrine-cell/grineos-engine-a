"""
Market Data Fetcher
===================
Pulls historical price and macro data from Yahoo Finance.
No API key required. Uses yfinance.

Driver proxies:
  Volatility  -> VIX index (^VIX)
  Growth      -> SPY 3-month price momentum
  Inflation   -> TIP/IEF ratio (TIPS vs nominal Treasuries)
  Liquidity   -> HYG/LQD ratio (HY credit vs IG credit)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# Default tickers for driver reconstruction
DRIVER_TICKERS = ["^VIX", "SPY", "TIP", "IEF", "HYG", "LQD"]

# Standard asset class ETF proxies for regime rebalancing
ASSET_CLASS_PROXIES = {
    "Equities":       "SPY",
    "Duration":       "TLT",
    "Commodities":    "DJP",
    "USD":            "UUP",
    "Credit (HY)":    "HYG",
    "Cash":           "SHV",
}


def fetch_prices(
    tickers: list,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    Compatible with yfinance >= 0.2.54.
    """
    if not end:
        end = datetime.today().strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        raw = pd.DataFrame()

    # Handle MultiIndex columns (newer yfinance returns (field, ticker) MultiIndex)
    if not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            prices = raw["Close"]
        elif "Adj Close" in level0:
            prices = raw["Adj Close"]
        else:
            prices = raw.iloc[:, :len(tickers) if isinstance(tickers, list) else 1]
    elif not raw.empty and "Close" in raw.columns:
        prices = raw[["Close"]]
        if isinstance(tickers, list) and len(tickers) == 1:
            prices.columns = tickers
    else:
        prices = raw

    # If download failed or returned empty, try one-by-one via Ticker.history
    if prices.empty or prices.isna().all().all():
        frames = {}
        ticker_list = tickers if isinstance(tickers, list) else [tickers]
        for ticker in ticker_list:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start=start, end=end, interval=interval, auto_adjust=True)
                if not hist.empty and "Close" in hist.columns:
                    frames[ticker] = hist["Close"]
            except Exception:
                pass
        prices = pd.DataFrame(frames) if frames else pd.DataFrame()

    prices = prices.dropna(how="all")
    return prices


def fetch_driver_data(start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch all data needed to reconstruct historical regime drivers.
    Returns a clean DataFrame with columns:
      vix, spy, tip, ief, hyg, lqd
    """
    prices = fetch_prices(DRIVER_TICKERS, start=start, end=end)
    prices.columns = [c.lower().replace("^", "") for c in prices.columns]
    prices = prices.ffill().dropna()
    return prices


def compute_driver_scores(driver_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw price series into normalised driver scores (-2 to +2).

    Volatility  = -VIX z-score (high VIX = negative volatility score)
    Growth      = SPY 63-day momentum z-score
    Inflation   = TIP/IEF ratio 63-day momentum z-score
    Liquidity   = HYG/LQD ratio z-score (credit conditions)
    """
    df = driver_df.copy()
    scores = pd.DataFrame(index=df.index)

    def zscore_clip(series, window=252):
        rolling_mean = series.rolling(window, min_periods=60).mean()
        rolling_std = series.rolling(window, min_periods=60).std()
        z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
        return z.clip(-2, 2)

    # Volatility: high VIX = bad, so negate
    if "vix" in df.columns:
        scores["volatility"] = zscore_clip(-df["vix"])

    # Growth: SPY price momentum (63-day return)
    if "spy" in df.columns:
        spy_mom = df["spy"].pct_change(63)
        scores["growth"] = zscore_clip(spy_mom)

    # Inflation: TIP/IEF ratio momentum (63-day)
    if "tip" in df.columns and "ief" in df.columns:
        tip_ratio = df["tip"] / df["ief"]
        tip_mom = tip_ratio.pct_change(63)
        scores["inflation"] = zscore_clip(tip_mom)

    # Liquidity: HYG/LQD ratio (tighter spreads = more liquidity)
    if "hyg" in df.columns and "lqd" in df.columns:
        credit_ratio = df["hyg"] / df["lqd"]
        scores["liquidity"] = zscore_clip(credit_ratio)

    scores = scores.dropna()
    return scores


def fetch_portfolio_prices(
    holdings: list,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical prices for portfolio holdings.
    holdings: list of {"ticker": str, "weight": float} dicts

    Falls back to SPY for any unresolvable ticker.
    Returns daily adjusted close prices.
    """
    tickers = list({h["ticker"] for h in holdings if h.get("ticker")})
    if not tickers:
        tickers = ["SPY"]

    prices = fetch_prices(tickers, start=start, end=end)

    # Fallback: replace any all-NaN column with SPY
    spy_prices = None
    for col in prices.columns:
        if prices[col].isna().all():
            if spy_prices is None:
                spy_prices = fetch_prices(["SPY"], start=start, end=end)["SPY"]
            prices[col] = spy_prices

    prices = prices.ffill().bfill()
    return prices
