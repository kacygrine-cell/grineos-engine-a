"""
Regime Classifier
=================
Maps four driver scores → one of six regimes + confidence.

Architecture:
  1. Rule-based classifier (fast, interpretable, default)
  2. Confidence = weighted agreement of rules
  3. FOG fires when no regime scores above the ambiguity threshold
  4. ML model hook: drop in a sklearn/torch model via classify_ml()

Regime decision logic (driver scores on [-2, +2]):
  SHOCK    : volatility > 1.2  OR  liquidity < -1.3  (crisis override)
  SURGE    : growth > 0.7  AND  inflation < 0.5  AND  liquidity > 0.4  AND  vol < -0.3
  CRUISE   : growth > 0.2  AND  inflation 0.2–1.1  AND  liquidity > -0.2  AND  vol < 0.6
  PRESSURE : inflation > 0.8  AND  growth < 0.4  (stagflation)
  SLIDE    : growth < -0.4  AND  liquidity < 0.0
  FOG      : ambiguous — no regime clears the confidence threshold
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import math

from .models import RegimeCode, TransitionProb


# ── Weights for composite regime score ───────────────────────────────────────
_WEIGHTS = {"growth": 0.30, "inflation": 0.25, "liquidity": 0.25, "volatility": 0.20}


def _weighted(drivers: Dict, key: str) -> float:
    return drivers[key].score


def classify(drivers: Dict) -> Tuple[RegimeCode, float, List[TransitionProb]]:
    """
    Return (regime_code, confidence_pct, transition_probs).
    drivers: Dict[str, DriverScore]
    """
    g  = drivers["growth"].score
    i  = drivers["inflation"].score
    lq = drivers["liquidity"].score
    v  = drivers["volatility"].score

    # ── Score each regime (0–1 raw affinity) ──────────────────────────────
    scores: Dict[str, float] = {}

    # SHOCK — crisis override: liquidity seizure or vol spike
    shock_raw = max(
        _sigmoid(v,    centre=1.3, steepness=3.0),
        _sigmoid(-lq,  centre=1.4, steepness=3.0),
    )
    scores["SHOCK"] = shock_raw

    # SURGE — best possible environment
    scores["SURGE"] = (
        _sigmoid(g,   centre=0.8,  steepness=2.5) *
        _sigmoid(-i,  centre=0.0,  steepness=2.0) *
        _sigmoid(lq,  centre=0.5,  steepness=2.5) *
        _sigmoid(-v,  centre=0.5,  steepness=2.5)
    )

    # CRUISE — late cycle, selective risk
    scores["CRUISE"] = (
        _sigmoid(g,  centre=0.4,  steepness=2.5) *
        _sigmoid(i,  centre=0.5,  steepness=2.0) *   # inflation mild-rising
        (1.0 - _sigmoid(-lq, centre=0.5, steepness=3.0)) *
        _sigmoid(-v, centre=0.0,  steepness=2.0)
    )

    # PRESSURE — stagflation: growth falling + inflation sticky
    scores["PRESSURE"] = (
        _sigmoid(i,  centre=0.9,  steepness=2.5) *
        _sigmoid(-g, centre=0.2,  steepness=2.5)
    )

    # SLIDE — recession: growth falling + liquidity tightening
    scores["SLIDE"] = (
        _sigmoid(-g,  centre=0.5, steepness=2.5) *
        _sigmoid(-lq, centre=0.3, steepness=2.5) *
        (1.0 - shock_raw)  # SHOCK dominates over SLIDE
    )

    # ── Normalise to probabilities ────────────────────────────────────────
    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}

    # ── FOG: if max probability below ambiguity threshold ────────────────
    AMBIGUITY_THRESHOLD = 0.38
    best_code = max(probs, key=probs.get)
    best_prob = probs[best_code]

    if best_prob < AMBIGUITY_THRESHOLD:
        regime = RegimeCode.FOG
        confidence = round(30.0 + best_prob * 40.0, 1)  # 30–50% range in FOG
        probs["FOG"] = 1.0 - best_prob
    else:
        regime = RegimeCode(best_code)
        confidence = round(50.0 + best_prob * 50.0, 1)   # 50–100%
        confidence = min(confidence, 98.0)

    # ── Build transition probability list ─────────────────────────────────
    transitions = [
        TransitionProb(regime=RegimeCode(k), probability=round(v, 3))
        for k, v in sorted(probs.items(), key=lambda x: -x[1])
    ]

    return regime, confidence, transitions


def _sigmoid(x: float, centre: float = 0.0, steepness: float = 2.0) -> float:
    """Smooth threshold function returning 0–1."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - centre)))


# ── ML model hook ─────────────────────────────────────────────────────────────
def classify_ml(drivers: Dict) -> Tuple[RegimeCode, float, List[TransitionProb]]:
    """
    Stub for ML-based classification.
    Replace the body with a trained model (sklearn, xgboost, pytorch, etc.)

    Example:
        features = [g, i, lq, v]
        probs = MODEL.predict_proba([features])[0]
        ...
    """
    # Fall back to rule-based until model is trained
    return classify(drivers)
