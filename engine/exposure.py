"""
Exposure Map Generator
======================
Maps regime → asset class signals for the GrineOS dashboard.

Each regime produces a deterministic baseline exposure map.
Driver scores are used to fine-tune signal strength at the margins.

Signal vocabulary:
  MAX OVERWEIGHT, OVERWEIGHT, NEUTRAL, FLOOR,
  ELEVATED, UNDERWEIGHT, MAX UNDERWEIGHT, MAXIMUM

Direction codes:
  "up2" = ++   "up" = +   "--" = neutral   "dn" = -   "dn2" = --
"""

from __future__ import annotations
from typing import Dict, List
from .models import RegimeCode, ExposureRow, ExposureMap
from datetime import datetime, timezone


# ── Baseline maps ─────────────────────────────────────────────────────────────
_BASE: Dict[str, List[tuple]] = {
    # (asset_class, signal, direction, magnitude, base_confidence)
    "SURGE": [
        ("Equities",    "MAX OVERWEIGHT",  "up2", "Strong",   91.0),
        ("Duration",    "UNDERWEIGHT",     "dn",  "Moderate", 78.0),
        ("Commodities", "OVERWEIGHT",      "up",  "Moderate", 74.0),
        ("USD",         "NEUTRAL",         "--",  "None",     62.0),
        ("Credit (HY)", "MAX OVERWEIGHT",  "up2", "Strong",   88.0),
        ("Cash",        "FLOOR",           "--",  "Minimum",  95.0),
    ],
    "CRUISE": [
        ("Equities",    "OVERWEIGHT",      "up",  "Moderate", 82.0),
        ("Duration",    "UNDERWEIGHT",     "dn",  "Light",    69.0),
        ("Commodities", "OVERWEIGHT",      "up",  "Light",    65.0),
        ("USD",         "NEUTRAL",         "--",  "None",     58.0),
        ("Credit (HY)", "NEUTRAL",         "--",  "None",     61.0),
        ("Cash",        "FLOOR",           "--",  "Minimum",  90.0),
    ],
    "PRESSURE": [
        ("Equities",    "UNDERWEIGHT",     "dn",  "Moderate", 76.0),
        ("Duration",    "MAX UNDERWEIGHT", "dn2", "Strong",   84.0),
        ("Commodities", "MAX OVERWEIGHT",  "up2", "Strong",   89.0),
        ("USD",         "OVERWEIGHT",      "up",  "Moderate", 71.0),
        ("Credit (HY)", "UNDERWEIGHT",     "dn",  "Light",    64.0),
        ("Cash",        "ELEVATED",        "up",  "Moderate", 87.0),
    ],
    "SLIDE": [
        ("Equities",    "UNDERWEIGHT",     "dn2", "Strong",   83.0),
        ("Duration",    "OVERWEIGHT",      "up",  "Moderate", 74.0),
        ("Commodities", "UNDERWEIGHT",     "dn",  "Light",    61.0),
        ("USD",         "OVERWEIGHT",      "up",  "Moderate", 77.0),
        ("Credit (HY)", "MAX UNDERWEIGHT", "dn2", "Strong",   86.0),
        ("Cash",        "ELEVATED",        "up2", "Strong",   92.0),
    ],
    "SHOCK": [
        ("Equities",    "MAX UNDERWEIGHT", "dn2", "Strong",   94.0),
        ("Duration",    "MAX OVERWEIGHT",  "up2", "Strong",   91.0),
        ("Commodities", "UNDERWEIGHT",     "dn",  "Moderate", 72.0),
        ("USD",         "MAX OVERWEIGHT",  "up2", "Strong",   88.0),
        ("Credit (HY)", "MAX UNDERWEIGHT", "dn2", "Strong",   96.0),
        ("Cash",        "MAXIMUM",         "up2", "Strong",   98.0),
    ],
    "FOG": [
        ("Equities",    "NEUTRAL",         "--",  "None",     54.0),
        ("Duration",    "NEUTRAL",         "--",  "None",     51.0),
        ("Commodities", "NEUTRAL",         "--",  "None",     49.0),
        ("USD",         "NEUTRAL",         "--",  "None",     52.0),
        ("Credit (HY)", "NEUTRAL",         "--",  "None",     50.0),
        ("Cash",        "ELEVATED",        "up",  "Light",    78.0),
    ],
}

# ── Driver-based confidence adjustments ──────────────────────────────────────
# Maps (regime, asset_class) → (driver_key, direction, max_adjustment)
# confidence adjusted by ±max_adjustment based on driver alignment
_ADJUSTMENTS = {
    ("SURGE",    "Equities"):    ("growth",    +1.0, 5.0),
    ("SURGE",    "Credit (HY)"): ("liquidity", +1.0, 5.0),
    ("PRESSURE", "Commodities"): ("inflation", +1.0, 4.0),
    ("PRESSURE", "Duration"):    ("inflation", +1.0, 4.0),
    ("SLIDE",    "Credit (HY)"): ("growth",    -1.0, 5.0),
    ("SHOCK",    "Cash"):        ("volatility",+1.0, 3.0),
}


def build_exposure(
    regime: RegimeCode,
    drivers: dict,
    regime_confidence: float,
) -> ExposureMap:
    base = _BASE.get(regime.value, _BASE["FOG"])
    rows: List[ExposureRow] = []

    for asset, signal, direction, magnitude, base_conf in base:
        # Apply driver-based confidence adjustment
        adj = _ADJUSTMENTS.get((regime.value, asset))
        conf = base_conf
        if adj:
            driver_key, driver_dir, max_adj = adj
            driver_score = drivers.get(driver_key)
            if driver_score:
                alignment = driver_dir * driver_score.score  # positive = aligned
                conf = min(99.0, max(5.0, conf + (alignment / 2.0) * max_adj))

        # Scale by overall regime confidence
        conf = round(conf * (regime_confidence / 100.0) * 1.05, 1)
        conf = min(99.0, max(5.0, conf))

        rows.append(ExposureRow(
            asset_class=asset,
            signal=signal,
            direction=direction,
            magnitude=magnitude,
            confidence=conf,
        ))

    return ExposureMap(
        regime=regime,
        subtitle=_subtitle(regime),
        rows=rows,
        timestamp=datetime.now(timezone.utc),
    )


def _subtitle(regime: RegimeCode) -> str:
    subtitles = {
        "SURGE":    "Risk Expansion",
        "CRUISE":   "Cycle Momentum",
        "PRESSURE": "Inflation Squeeze",
        "SLIDE":    "Growth Breakdown",
        "SHOCK":    "Liquidity Break",
        "FOG":      "Regime Uncertainty",
    }
    return subtitles.get(regime.value, "")
