"""
Narrative Engine
================
Generates the one-line regime brief for GrineOS.

Format: [REGIME] regime [status] — [primary driver] — [action].

Each regime has a set of narrative templates. The engine selects
the template that best matches the current driver configuration.

Horizon logic:
  - Confidence rising + persistence > 7d  → Medium
  - Confidence falling or new signal       → Short
  - FOG always → Short

Risk flag logic:
  One sentence identifying the single driver most likely to
  trigger a regime change.
"""

from __future__ import annotations
from typing import Dict
from datetime import datetime, timezone
from .models import RegimeCode, Narrative


# ── Narrative templates ───────────────────────────────────────────────────────
_TEMPLATES: Dict[str, list] = {
    "SURGE": [
        "SURGE regime confirmed — growth is accelerating with liquidity intact. Risk should be fully deployed.",
        "SURGE regime active — all four drivers aligned. Maximum risk deployment is warranted.",
        "SURGE regime — volatility suppressed, growth running. Overweight equities and credit now.",
    ],
    "CRUISE": [
        "CRUISE regime — momentum intact but narrowing. Stay invested, raise quality, trim duration.",
        "CRUISE regime confirmed — late-cycle conditions. Selective risk with quality bias.",
        "CRUISE regime active — growth positive but inflation rising. Tighten selection, hold positions.",
    ],
    "PRESSURE": [
        "PRESSURE regime — 60/40 is structurally challenged. Real assets and short duration are the only clean trades.",
        "PRESSURE regime confirmed — inflation sticky, growth fading. Cut bonds, add commodities.",
        "PRESSURE regime active — stagflation conditions. Inflation hedges are the priority.",
    ],
    "SLIDE": [
        "SLIDE regime — cycle is breaking down. De-risk now. Defensives and quality bonds only.",
        "SLIDE regime confirmed — growth collapsing, spreads widening. Exit high beta immediately.",
        "SLIDE regime active — recession dynamics in play. Raise cash, rotate to quality.",
    ],
    "SHOCK": [
        "SHOCK regime active — liquidity has broken. Capital preservation only. No exceptions.",
        "SHOCK regime — volatility has spiked, liquidity seized. Cash and sovereigns are the only shelter.",
        "SHOCK regime confirmed — crisis conditions. Review every position. Act fast.",
    ],
    "FOG": [
        "FOG regime — signals are contradictory. Hold balanced positions. Wait for confirmation before adding risk.",
        "FOG regime active — high uncertainty. Reduce conviction sizing. Do not force a trade.",
        "FOG regime confirmed — mixed drivers. Neutral allocation. Let the signals resolve.",
    ],
}

_RISK_FLAGS: Dict[str, list] = {
    "SURGE":    [
        "Watch for inflation re-acceleration above 3.5% — would trigger PRESSURE.",
        "Watch for liquidity tightening by the Fed — would shift toward CRUISE.",
        "Watch for volatility breakout above VIX 22 — early warning of CRUISE.",
    ],
    "CRUISE":   [
        "Watch for liquidity tightening — would accelerate transition to SLIDE or PRESSURE.",
        "Watch for inflation persistence — would push toward PRESSURE.",
        "Watch for growth disappointment — would shift toward SLIDE.",
    ],
    "PRESSURE": [
        "Watch for growth collapse — would flip rapidly into SLIDE or SHOCK.",
        "Watch for inflation break lower — would shift toward CRUISE or SLIDE.",
        "Watch for credit spread widening — would escalate to SHOCK.",
    ],
    "SLIDE":    [
        "Watch for liquidity seizure — any credit market freeze would escalate to SHOCK.",
        "Watch for central bank pivot — liquidity injection would trigger FOG or CRUISE.",
        "Watch for inflation re-emergence — would create PRESSURE conditions.",
    ],
    "SHOCK":    [
        "Watch for central bank intervention — emergency liquidity injection would trigger FOG or SLIDE.",
        "Watch for volatility normalisation below VIX 25 — would open path to SLIDE.",
        "Watch for credit market re-opening — key signal for regime stabilisation.",
    ],
    "FOG":      [
        "Watch all four drivers simultaneously — whichever resolves first determines the next regime.",
        "Watch growth vs inflation divergence — will determine SURGE/CRUISE or PRESSURE path.",
        "Watch liquidity conditions — tightening pushes to SLIDE, easing opens SURGE path.",
    ],
}


def generate(
    regime: RegimeCode,
    drivers: dict,
    confidence: float,
    confidence_delta: float,
    persistence_days: int,
) -> Narrative:
    # Select template based on driver alignment
    templates = _TEMPLATES[regime.value]
    idx = _select_template(regime, drivers)
    text = templates[idx % len(templates)]

    # Horizon
    if regime == RegimeCode.FOG:
        horizon = "Short"
    elif confidence >= 80.0 and confidence_delta >= 0 and persistence_days >= 7:
        horizon = "Medium"
    else:
        horizon = "Short"

    # Risk flag — pick based on weakest driver
    flags = _RISK_FLAGS[regime.value]
    flag_idx = _weakest_driver_index(regime, drivers)
    risk_flag = flags[flag_idx % len(flags)]

    return Narrative(
        regime=regime,
        subtitle=_subtitle(regime),
        text=text,
        horizon=horizon,
        risk_flag=risk_flag,
        timestamp=datetime.now(timezone.utc),
    )


def _select_template(regime: RegimeCode, drivers: dict) -> int:
    """Choose the most contextually appropriate template."""
    g  = drivers.get("growth",    type("", (), {"score": 0.0})()).score
    i  = drivers.get("inflation", type("", (), {"score": 0.0})()).score
    lq = drivers.get("liquidity", type("", (), {"score": 0.0})()).score
    v  = drivers.get("volatility",type("", (), {"score": 0.0})()).score

    if regime == RegimeCode.SURGE:
        if g > 1.2: return 0
        if lq > 1.0: return 1
        return 2
    if regime == RegimeCode.SHOCK:
        if v > 1.5: return 0
        if lq < -1.5: return 1
        return 2
    if regime == RegimeCode.PRESSURE:
        if i > 1.2: return 0
        return 1
    return 0


def _weakest_driver_index(regime: RegimeCode, drivers: dict) -> int:
    """Index based on which driver is least aligned with the regime ideal."""
    g  = drivers.get("growth",    type("", (), {"score": 0.0})()).score
    i  = drivers.get("inflation", type("", (), {"score": 0.0})()).score
    lq = drivers.get("liquidity", type("", (), {"score": 0.0})()).score
    v  = drivers.get("volatility",type("", (), {"score": 0.0})()).score

    # Simple priority: use the most extreme adverse signal
    adverse = {"growth": -g, "inflation": i, "liquidity": -lq, "volatility": v}
    worst = max(adverse, key=adverse.get)
    mapping = {"growth": 0, "inflation": 1, "liquidity": 2, "volatility": 0}
    return mapping.get(worst, 0)


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
