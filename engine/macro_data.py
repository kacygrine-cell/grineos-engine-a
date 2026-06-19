"""
Engine B — Macro Data Connector
================================
Pulls real macroeconomic data from FRED (Federal Reserve Economic Data).
All series are free, no rate limits for reasonable use.

Series used:
  CPIAUCSL  — Consumer Price Index (monthly) — inflation
  FEDFUNDS  — Fed Funds Effective Rate (monthly) — monetary policy
  DGS2      — 2-Year Treasury yield (daily) — short rates
  DGS10     — 10-Year Treasury yield (daily) — long rates / risk sentiment
  INDPRO    — Industrial Production Index (monthly) — growth proxy
  M2SL      — M2 Money Supply (weekly) — liquidity proxy

Cache: 6 hours (macro data moves slowly)
"""

import os
import time
import math
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_cache: Dict = {}
CACHE_TTL = 21600  # 6 hours


def _get_fred_client():
    """Return a fredapi client, raising if key not configured."""
    from fredapi import Fred
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY not configured in environment.")
    return Fred(api_key=key)


def _zscore(values: list, value: float, clip: float = 2.0) -> float:
    """Z-score of value vs list, clipped to [-clip, +clip]."""
    if len(values) < 6:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0
    return max(-clip, min(clip, (value - mean) / std))


def fetch_macro_signals() -> dict:
    """
    Fetch and compute all Engine B macro signals from FRED.
    Returns a dict of raw values and normalised scores.

    Cached for 6 hours — macro data moves slowly.
    """
    now = time.time()
    if _cache.get("macro") and (now - _cache.get("macro_ts", 0)) < CACHE_TTL:
        return _cache["macro"]

    result = {
        "cpi_yoy": None,          # CPI year-over-year %
        "cpi_momentum": None,     # CPI 3-month acceleration
        "fed_rate": None,         # Current Fed Funds rate
        "fed_rate_change_3m": None, # Change in Fed rate over 3 months
        "yield_curve": None,      # 10Y - 2Y spread
        "indpro_momentum": None,  # Industrial production 3-month momentum
        "m2_growth": None,        # M2 growth rate (liquidity)
        "scores": {               # Normalised scores for ensemble
            "cpi_pressure": 0.0,
            "policy_tightening": 0.0,
            "yield_curve_signal": 0.0,
            "growth_macro": 0.0,
            "liquidity_macro": 0.0,
        },
        "status": "ok",
    }

    try:
        fred = _get_fred_client()
        import pandas as pd

        # ── CPI ──────────────────────────────────────────────────────────────
        try:
            cpi = fred.get_series("CPIAUCSL", observation_start="2018-01-01")
            cpi = cpi.dropna()
            if len(cpi) >= 14:
                # YoY
                result["cpi_yoy"] = round(
                    float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100), 2
                )
                # 3-month momentum (acceleration)
                cpi_3m_ago = float(cpi.iloc[-4])
                cpi_now = float(cpi.iloc[-1])
                cpi_mom = ((cpi_now / cpi_3m_ago) - 1) * 100 * 4  # annualised
                result["cpi_momentum"] = round(cpi_mom, 2)
                # Score: high CPI = PRESSURE signal
                # >6% annualised = strong pressure
                result["scores"]["cpi_pressure"] = round(
                    max(-2, min(2, (cpi_mom - 2.5) / 2.0)), 3
                )
        except Exception as e:
            logger.warning(f"CPI fetch failed: {e}")

        # ── Fed Funds Rate ────────────────────────────────────────────────────
        try:
            fed = fred.get_series("FEDFUNDS", observation_start="2018-01-01")
            fed = fed.dropna()
            if len(fed) >= 4:
                result["fed_rate"] = round(float(fed.iloc[-1]), 3)
                change_3m = float(fed.iloc[-1]) - float(fed.iloc[-4])
                result["fed_rate_change_3m"] = round(change_3m, 3)
                # Score: hiking = negative liquidity/growth signal
                # Large hike (>0.75pp in 3m) = aggressive tightening
                result["scores"]["policy_tightening"] = round(
                    max(-2, min(2, change_3m / 0.5)), 3
                )
        except Exception as e:
            logger.warning(f"Fed rate fetch failed: {e}")

        # ── Yield Curve (10Y - 2Y) ────────────────────────────────────────────
        try:
            y10 = fred.get_series("DGS10", observation_start="2018-01-01").dropna()
            y2 = fred.get_series("DGS2", observation_start="2018-01-01").dropna()
            if len(y10) >= 20 and len(y2) >= 20:
                spread_now = float(y10.iloc[-1]) - float(y2.iloc[-1])
                result["yield_curve"] = round(spread_now, 3)
                # Historical spread values
                common = y10.index.intersection(y2.index)
                spreads = (y10.loc[common] - y2.loc[common]).dropna()
                spread_list = spreads.tolist()
                # Score: inversion = recession signal = negative growth/liquidity
                result["scores"]["yield_curve_signal"] = round(
                    _zscore(spread_list, spread_now), 3
                )
        except Exception as e:
            logger.warning(f"Yield curve fetch failed: {e}")

        # ── Industrial Production ─────────────────────────────────────────────
        try:
            indpro = fred.get_series("INDPRO", observation_start="2018-01-01").dropna()
            if len(indpro) >= 6:
                mom_3m = float(indpro.iloc[-1] / indpro.iloc[-4] - 1) * 100
                result["indpro_momentum"] = round(mom_3m, 3)
                all_mom = [
                    float(indpro.iloc[i] / indpro.iloc[i - 3] - 1) * 100
                    for i in range(3, len(indpro))
                ]
                result["scores"]["growth_macro"] = round(
                    _zscore(all_mom, mom_3m), 3
                )
        except Exception as e:
            logger.warning(f"INDPRO fetch failed: {e}")

        # ── M2 Money Supply ───────────────────────────────────────────────────
        try:
            m2 = fred.get_series("M2SL", observation_start="2018-01-01").dropna()
            if len(m2) >= 14:
                m2_growth_yoy = float(m2.iloc[-1] / m2.iloc[-13] - 1) * 100
                result["m2_growth"] = round(m2_growth_yoy, 2)
                # Positive M2 growth = liquidity supportive
                all_m2g = [
                    float(m2.iloc[i] / m2.iloc[i - 13] - 1) * 100
                    for i in range(13, len(m2))
                ]
                result["scores"]["liquidity_macro"] = round(
                    _zscore(all_m2g, m2_growth_yoy), 3
                )
        except Exception as e:
            logger.warning(f"M2 fetch failed: {e}")

    except Exception as e:
        result["status"] = f"error: {str(e)}"
        logger.error(f"Macro data fetch failed: {e}")

    _cache["macro"] = result
    _cache["macro_ts"] = time.time()
    return result


def get_macro_driver_scores(macro: dict) -> dict:
    """
    Convert raw macro signals into Engine A-compatible driver scores.
    These will be blended with Engine A's price-based scores in the ensemble.

    Returns dict with keys: growth, inflation, liquidity, volatility
    Each value in [-2.0, +2.0]
    """
    scores = macro.get("scores", {})

    # Growth: industrial production momentum
    growth = scores.get("growth_macro", 0.0)

    # Inflation: CPI pressure score (high CPI = high positive inflation score)
    inflation = scores.get("cpi_pressure", 0.0)

    # Liquidity: blend of yield curve signal, policy tightening, M2
    # Inversion and hiking = tightening = negative liquidity
    yield_sig = scores.get("yield_curve_signal", 0.0)
    policy_sig = -scores.get("policy_tightening", 0.0)  # hiking = negative
    m2_sig = scores.get("liquidity_macro", 0.0)
    liquidity = round((yield_sig * 0.4 + policy_sig * 0.4 + m2_sig * 0.2), 3)
    liquidity = max(-2.0, min(2.0, liquidity))

    # Volatility: not well-captured by FRED macro — keep 0 (defer to Engine A)
    volatility = 0.0

    return {
        "growth": round(growth, 3),
        "inflation": round(inflation, 3),
        "liquidity": round(liquidity, 3),
        "volatility": round(volatility, 3),
    }
