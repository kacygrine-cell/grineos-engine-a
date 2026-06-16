"""
Historical Regime Reconstructor
================================
Runs Engine A's regime classifier backward over historical driver scores
to produce a daily regime series.

This is the bridge between real market data and the backtest engine.
"""

import pandas as pd
import numpy as np
from typing import Optional
from functools import lru_cache

from engine.data.market_data import fetch_driver_data, compute_driver_scores


# Regime classification thresholds
# Mirror the logic in engine/classifier.py but applied to time series
REGIME_RULES = {
    "SURGE": {
        "growth":     (0.5, 2.0),
        "inflation":  (-2.0, 0.5),
        "liquidity":  (0.0, 2.0),
        "volatility": (-2.0, 0.0),   # low VIX
    },
    "CRUISE": {
        "growth":     (0.0, 1.5),
        "inflation":  (-0.5, 1.5),
        "liquidity":  (-0.5, 2.0),
        "volatility": (-2.0, 0.3),
    },
    "PRESSURE": {
        "growth":     (-1.0, 0.5),
        "inflation":  (0.5, 2.0),
        "liquidity":  (-2.0, 0.5),
        "volatility": (-0.5, 1.0),
    },
    "SLIDE": {
        "growth":     (-2.0, -0.3),
        "inflation":  (-2.0, 1.0),
        "liquidity":  (-2.0, 0.0),
        "volatility": (0.3, 2.0),
    },
    "SHOCK": {
        "growth":     (-2.0, -0.5),
        "inflation":  (-2.0, 2.0),
        "liquidity":  (-2.0, -0.5),
        "volatility": (1.0, 2.0),
    },
    "FOG": {
        "growth":     (-0.5, 0.5),
        "inflation":  (-0.5, 0.5),
        "liquidity":  (-0.5, 0.5),
        "volatility": (-0.5, 0.5),
    },
}

# Ordered by priority (more extreme regimes take precedence)
REGIME_PRIORITY = ["SHOCK", "SLIDE", "SURGE", "PRESSURE", "CRUISE", "FOG"]


def _classify_row(growth: float, inflation: float, liquidity: float, volatility: float) -> str:
    """Classify a single date's drivers into a regime."""
    scores = {
        "growth": growth,
        "inflation": inflation,
        "liquidity": liquidity,
        "volatility": volatility,
    }

    best_regime = "FOG"
    best_score = 0

    for regime in REGIME_PRIORITY:
        rules = REGIME_RULES[regime]
        match_count = 0
        for driver, (low, high) in rules.items():
            if low <= scores.get(driver, 0) <= high:
                match_count += 1

        # Need at least 3 out of 4 drivers to match
        if match_count >= 3 and match_count > best_score:
            best_score = match_count
            best_regime = regime

    return best_regime


def reconstruct_regime_history(
    start: str,
    end: Optional[str] = None,
    smoothing_days: int = 5,
) -> pd.Series:
    """
    Reconstruct daily regime history by:
    1. Fetching historical driver proxy data from Yahoo Finance
    2. Computing normalised driver scores
    3. Running the classifier on each date

    smoothing_days: apply a mode filter over N days to reduce noise

    Returns a pd.Series indexed by date with regime codes.
    """
    driver_df = fetch_driver_data(start=start, end=end)
    scores_df = compute_driver_scores(driver_df)

    regimes = pd.Series(index=scores_df.index, dtype=str)

    for date, row in scores_df.iterrows():
        regimes[date] = _classify_row(
            growth=row.get("growth", 0),
            inflation=row.get("inflation", 0),
            liquidity=row.get("liquidity", 0),
            volatility=row.get("volatility", 0),
        )

    # Smooth: rolling mode to reduce single-day noise
    if smoothing_days > 1:
        def rolling_mode(series, window):
            result = series.copy()
            for i in range(window - 1, len(series)):
                window_vals = series.iloc[i - window + 1:i + 1]
                result.iloc[i] = window_vals.mode().iloc[0]
            return result

        regimes = rolling_mode(regimes, smoothing_days)

    return regimes


def get_regime_change_dates(regime_series: pd.Series) -> pd.DataFrame:
    """
    Extract dates where the regime changed.
    Returns a DataFrame with columns: date, from_regime, to_regime
    """
    changes = []
    prev_regime = None
    for date, regime in regime_series.items():
        if prev_regime is not None and regime != prev_regime:
            changes.append({
                "date": date,
                "from_regime": prev_regime,
                "to_regime": regime,
            })
        prev_regime = regime

    return pd.DataFrame(changes)
