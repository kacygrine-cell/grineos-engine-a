"""
Engine B — Ensemble Classifier
================================
Blends Engine A (price-based, fast, reactive) with Engine B macro signals
(FRED data, leading, structural) to produce a more accurate regime reading.

Blend weights:
  growth     — 50% Engine A (SPY momentum) + 50% Engine B (INDPRO)
  inflation  — 30% Engine A (TIP/IEF)      + 70% Engine B (CPI)
  liquidity  — 50% Engine A (HYG/LQD)      + 50% Engine B (yield curve + M2 + policy)
  volatility — 100% Engine A (VIX)          — price-based only for vol

Rationale:
  Inflation and liquidity are where macro leads most reliably.
  CPI and Fed policy signal PRESSURE weeks/months before price data deteriorates.
  Growth uses both equally — INDPRO and SPY momentum are complementary.
  Volatility has no good macro leading indicator — keep price-based.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Blend weights: (engine_a_weight, engine_b_weight)
BLEND_WEIGHTS = {
    "growth":     (0.5, 0.5),
    "inflation":  (0.3, 0.7),   # Macro leads on inflation
    "liquidity":  (0.5, 0.5),
    "volatility": (1.0, 0.0),   # Price-based only
}


def blend_drivers(
    engine_a_drivers: dict,
    engine_b_drivers: dict,
) -> dict:
    """
    Blend Engine A and Engine B driver scores.

    engine_a_drivers: {"growth": float, "inflation": float, ...}
    engine_b_drivers: {"growth": float, "inflation": float, ...}

    Returns blended dict with same keys.
    """
    blended = {}
    for driver, (wa, wb) in BLEND_WEIGHTS.items():
        a_score = engine_a_drivers.get(driver, 0.0)
        b_score = engine_b_drivers.get(driver, 0.0)
        blended[driver] = round(a_score * wa + b_score * wb, 3)
    return blended


def run_engine_b(engine_a_summary) -> dict:
    """
    Run Engine B by blending Engine A with FRED macro data.

    Returns a complete Engine B result dict including:
      - regime: the blended regime code
      - drivers_a: Engine A raw drivers
      - drivers_b: Engine B macro drivers
      - drivers_blended: the final blended scores
      - divergence: whether A and B disagree on regime
      - macro_signals: raw FRED values
      - color, subtitle, instinct from regime meta
    """
    from engine.macro_data import fetch_macro_signals, get_macro_driver_scores

    try:
        # Fetch macro signals from FRED
        macro = fetch_macro_signals()
        drivers_b = get_macro_driver_scores(macro)
    except Exception as e:
        logger.error(f"Engine B macro fetch failed: {e}")
        macro = {"status": f"error: {e}", "scores": {}}
        drivers_b = {"growth": 0.0, "inflation": 0.0, "liquidity": 0.0, "volatility": 0.0}

    # Extract Engine A driver scores
    state = engine_a_summary.state
    raw_drivers = state.drivers or {}

    def get_score(key):
        d = raw_drivers.get(key)
        if d is None:
            return 0.0
        return float(d.score) if hasattr(d, "score") else float(d)

    drivers_a = {
        "growth":     get_score("growth"),
        "inflation":  get_score("inflation"),
        "liquidity":  get_score("liquidity"),
        "volatility": get_score("volatility"),
    }

    # Blend
    blended = blend_drivers(drivers_a, drivers_b)

    # Run regime classifier on blended scores
    # Import the engine's simulate method
    return {
        "drivers_a": drivers_a,
        "drivers_b": drivers_b,
        "drivers_blended": blended,
        "macro_signals": {
            "cpi_yoy": macro.get("cpi_yoy"),
            "cpi_momentum": macro.get("cpi_momentum"),
            "fed_rate": macro.get("fed_rate"),
            "fed_rate_change_3m": macro.get("fed_rate_change_3m"),
            "yield_curve": macro.get("yield_curve"),
            "indpro_momentum": macro.get("indpro_momentum"),
            "m2_growth": macro.get("m2_growth"),
        },
        "macro_status": macro.get("status", "ok"),
        "blend_weights": BLEND_WEIGHTS,
    }


def get_divergence_warning(regime_a: str, regime_b: str) -> Optional[str]:
    """
    Return a warning message if Engine A and Engine B disagree significantly.
    Regimes are ordered by risk level — large gaps indicate macro stress.
    """
    RISK_ORDER = {"SHOCK": 0, "SLIDE": 1, "PRESSURE": 2, "FOG": 3, "CRUISE": 4, "SURGE": 5}
    a_risk = RISK_ORDER.get(regime_a, 3)
    b_risk = RISK_ORDER.get(regime_b, 3)
    gap = abs(a_risk - b_risk)

    if gap == 0:
        return None
    if gap == 1:
        return f"Minor divergence: Engine A reads {regime_a}, macro signals suggest {regime_b}. Monitor."
    if gap == 2:
        return f"Significant divergence: Engine A reads {regime_a} but macro signals read {regime_b}. Macro often leads — consider reducing risk."
    return f"Major divergence: Engine A reads {regime_a} but macro signals read {regime_b}. High uncertainty. Defensive positioning warranted."
