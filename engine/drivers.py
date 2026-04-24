"""
Driver Scoring Engine
=====================
Converts raw market data into normalised driver scores (-2 to +2).

Four drivers:
  - Growth     : GDP momentum, PMI, earnings revisions
  - Inflation  : CPI, PPI, inflation expectations
  - Liquidity  : Credit spreads, M2 growth, central bank posture
  - Volatility : VIX, MOVE index, realised vol

Each raw input is normalised against a rolling z-score window,
then clamped to [-2, +2] and labelled.

Connector stubs in engine/data/connectors.py provide the raw values.
Synthetic data is used when live data is unavailable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math

from .models import DriverScore


# ── Label maps ────────────────────────────────────────────────────────────────
GROWTH_LABELS = {
    (1.0, 2.1):  "Accelerating",
    (0.3, 1.0):  "Positive",
    (-0.3, 0.3): "Neutral",
    (-1.0, -0.3):"Decelerating",
    (-2.1, -1.0):"Stalling",
}

INFLATION_LABELS = {
    (1.0, 2.1):  "Surging",
    (0.3, 1.0):  "Rising",
    (-0.3, 0.3): "Contained",
    (-1.0, -0.3):"Falling",
    (-2.1, -1.0):"Deflationary",
}

LIQUIDITY_LABELS = {
    (1.0, 2.1):  "Abundant",
    (0.3, 1.0):  "Ample",
    (-0.3, 0.3): "Neutral",
    (-1.0, -0.3):"Tightening",
    (-2.1, -1.0):"Stressed",
}

VOLATILITY_LABELS = {
    (-2.1, -1.0):"Suppressed",
    (-1.0, -0.3):"Low",
    (-0.3, 0.3): "Moderate",
    (0.3, 1.0):  "Elevated",
    (1.0, 2.1):  "Spiking",
}


def _label(score: float, label_map: dict) -> str:
    for (lo, hi), label in label_map.items():
        if lo <= score < hi:
            return label
    return "Neutral"


def _trend(current: float, previous: float) -> str:
    diff = current - previous
    if diff > 0.15:  return "rising"
    if diff < -0.15: return "falling"
    return "stable"


def _confidence(score: float, target_range: Tuple[float, float]) -> float:
    """
    Confidence = how clearly the score falls inside the target range.
    Returns 0–100.
    """
    lo, hi = target_range
    centre = (lo + hi) / 2
    half_width = (hi - lo) / 2 if (hi - lo) > 0 else 1.0
    dist = abs(score - centre)
    # Gaussian-style confidence: max at centre, ~50% at edge
    conf = 100.0 * math.exp(-0.5 * (dist / half_width) ** 2)
    return round(min(max(conf, 5.0), 99.0), 1)


def score_drivers(
    raw: Dict[str, float],
    previous: Dict[str, float],
) -> Dict[str, DriverScore]:
    """
    Convert raw driver inputs into DriverScore objects.

    raw / previous keys: "growth", "inflation", "liquidity", "volatility"
    Values are already normalised z-scores on [-2, +2].
    """
    def clamp(v: float) -> float:
        return max(-2.0, min(2.0, v))

    g  = clamp(raw.get("growth",    0.0))
    i  = clamp(raw.get("inflation", 0.0))
    lq = clamp(raw.get("liquidity", 0.0))
    v  = clamp(raw.get("volatility",0.0))

    pg  = clamp(previous.get("growth",    g))
    pi  = clamp(previous.get("inflation", i))
    plq = clamp(previous.get("liquidity", lq))
    pv  = clamp(previous.get("volatility",v))

    return {
        "growth": DriverScore(
            name="Growth",
            score=round(g, 3),
            label=_label(g, GROWTH_LABELS),
            confidence=_confidence(g, (0.5, 2.0)),
            trend=_trend(g, pg),
        ),
        "inflation": DriverScore(
            name="Inflation",
            score=round(i, 3),
            label=_label(i, INFLATION_LABELS),
            confidence=_confidence(i, (-0.5, 0.5)),
            trend=_trend(i, pi),
        ),
        "liquidity": DriverScore(
            name="Liquidity",
            score=round(lq, 3),
            label=_label(lq, LIQUIDITY_LABELS),
            confidence=_confidence(lq, (0.5, 2.0)),
            trend=_trend(lq, plq),
        ),
        "volatility": DriverScore(
            name="Volatility",
            score=round(v, 3),
            label=_label(v, VOLATILITY_LABELS),
            confidence=_confidence(v, (-2.0, -0.5)),
            trend=_trend(v, pv),
        ),
    }
