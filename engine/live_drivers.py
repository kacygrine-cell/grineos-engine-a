"""
Live Driver Computation
=======================
Pulls real market data from Yahoo Finance and computes normalised
driver scores (-2.0 to +2.0) for the regime classifier.

Drivers:
  growth     -> SPY 63-day price momentum (z-scored vs 1Y rolling)
  inflation  -> TIP/IEF ratio momentum (TIPS vs nominal Treasuries)
  liquidity  -> HYG/LQD ratio z-score (HY credit vs IG credit)
  volatility -> Negative VIX z-score (high VIX = negative vol driver)

All scores are clipped to [-2, +2] matching the classifier input range.
Results are cached for 15 minutes to avoid hammering Yahoo Finance.
"""

import time
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache: store (scores_dict, timestamp)
_cache: dict = {"data": None, "ts": 0}
CACHE_TTL = 900  # 15 minutes


def _zscore_clip(values: list, value: float, clip: float = 2.0) -> float:
    """Compute z-score of value vs list, clipped to [-clip, +clip]."""
    if len(values) < 10:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    z = (value - mean) / std
    return max(-clip, min(clip, z))


def fetch_live_drivers() -> dict:
    """
    Fetch live driver scores from Yahoo Finance.
    Returns dict with keys: growth, inflation, liquidity, volatility
    Each value is a float in [-2.0, +2.0].
    Falls back to neutral (0.0) on any failure.
    """
    now = time.time()
    if _cache["data"] and (now - _cache["ts"]) < CACHE_TTL:
        return _cache["data"]

    scores = {"growth": 0.0, "inflation": 0.0, "liquidity": 0.0, "volatility": 0.0}

    try:
        import yfinance as yf
        import pandas as pd

        # Fetch all tickers at once - 252 trading days
        tickers = ["SPY", "^VIX", "TIP", "IEF", "HYG", "LQD"]
        raw = yf.download(
            tickers, period="1y", interval="1d",
            auto_adjust=True, progress=False, threads=True
        )

        # Extract close prices - handle MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            else:
                prices = raw.iloc[:, :len(tickers)]
        else:
            prices = raw

        prices = prices.ffill().dropna(how="all")

        if prices.empty or len(prices) < 20:
            logger.warning("Live data: insufficient price history")
            return scores

        def get_series(ticker):
            ticker_clean = ticker.replace("^", "")
            # Try both with and without caret
            for col in [ticker, ticker_clean, ticker.upper()]:
                if col in prices.columns:
                    return prices[col].dropna()
            return pd.Series(dtype=float)

        # ── Growth: SPY 63-day momentum z-score ───────────────────────────
        spy = get_series("SPY")
        if len(spy) >= 70:
            mom_series = spy.pct_change(63).dropna()
            if len(mom_series) >= 10:
                current_mom = float(mom_series.iloc[-1])
                historical = mom_series.tolist()
                scores["growth"] = round(_zscore_clip(historical, current_mom), 3)

        # ── Volatility: negative VIX z-score ──────────────────────────────
        vix = get_series("^VIX")
        if len(vix) >= 20:
            vix_list = vix.tolist()
            current_vix = float(vix.iloc[-1])
            # Negate: high VIX = bad = negative vol driver
            scores["volatility"] = round(-_zscore_clip(vix_list, current_vix), 3)

        # ── Inflation: TIP/IEF ratio 63-day momentum ───────────────────────
        tip = get_series("TIP")
        ief = get_series("IEF")
        if len(tip) >= 70 and len(ief) >= 70:
            aligned = pd.concat([tip, ief], axis=1, join="inner").dropna()
            aligned.columns = ["TIP", "IEF"]
            ratio = aligned["TIP"] / aligned["IEF"]
            inf_mom = ratio.pct_change(63).dropna()
            if len(inf_mom) >= 10:
                current = float(inf_mom.iloc[-1])
                scores["inflation"] = round(_zscore_clip(inf_mom.tolist(), current), 3)

        # ── Liquidity: HYG/LQD ratio z-score ──────────────────────────────
        hyg = get_series("HYG")
        lqd = get_series("LQD")
        if len(hyg) >= 20 and len(lqd) >= 20:
            aligned2 = pd.concat([hyg, lqd], axis=1, join="inner").dropna()
            aligned2.columns = ["HYG", "LQD"]
            cr = aligned2["HYG"] / aligned2["LQD"]
            cr_list = cr.tolist()
            current_cr = float(cr.iloc[-1])
            scores["liquidity"] = round(_zscore_clip(cr_list, current_cr), 3)

        logger.info(f"Live drivers computed: {scores}")

    except Exception as e:
        logger.error(f"Live driver fetch failed: {e}")
        # Return neutral scores on failure
        scores = {"growth": 0.0, "inflation": 0.0, "liquidity": 0.0, "volatility": 0.0}

    _cache["data"] = scores
    _cache["ts"] = time.time()
    return scores


def get_driver_labels(scores: dict) -> dict:
    """Convert raw scores to human-readable labels matching Engine A format."""
    def label(score, positive_label, negative_label, neutral_label="Neutral"):
        if score > 0.5:
            return positive_label
        elif score < -0.5:
            return negative_label
        return neutral_label

    return {
        "growth":     label(scores["growth"],     "Accelerating", "Decelerating", "Stable"),
        "inflation":  label(scores["inflation"],  "Rising",       "Falling",      "Contained"),
        "liquidity":  label(scores["liquidity"],  "Abundant",     "Tightening",   "Neutral"),
        "volatility": label(scores["volatility"], "Suppressed",   "Elevated",     "Moderate"),
    }
