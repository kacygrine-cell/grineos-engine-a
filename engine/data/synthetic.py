"""
Synthetic Data Generator
========================
Generates realistic driver time-series for demo/dev use.

Produces plausible regime cycles based on historical macro patterns:
  - Full cycle: ~18 months SURGE → CRUISE → PRESSURE → SLIDE → SHOCK → FOG → SURGE
  - Each regime has characteristic driver patterns with noise
  - Confidence builds gradually, then decays at transitions

Used when live data connectors are unavailable or in demo mode.
"""

from __future__ import annotations
import math
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple


# ── Driver patterns per regime phase ──────────────────────────────────────────
# (growth_centre, inflation_centre, liquidity_centre, volatility_centre)
REGIME_PATTERNS: Dict[str, Tuple[float, float, float, float]] = {
    "SURGE":    ( 1.2,  -0.3,  1.1,  -1.3),
    "CRUISE":   ( 0.6,   0.5,  0.4,  -0.3),
    "PRESSURE": (-0.2,   1.3, -0.3,   0.5),
    "SLIDE":    (-1.0,   0.2, -0.8,   0.9),
    "SHOCK":    (-1.5,  -0.1, -1.8,   1.8),
    "FOG":      ( 0.0,   0.0,  0.0,   0.2),
}

# Canonical cycle sequence
CYCLE = ["SURGE", "CRUISE", "PRESSURE", "SLIDE", "SHOCK", "FOG"]

# Duration range (days) per regime
DURATION_RANGE: Dict[str, Tuple[int, int]] = {
    "SURGE":    (60, 140),
    "CRUISE":   (45, 100),
    "PRESSURE": (30,  90),
    "SLIDE":    (30,  75),
    "SHOCK":    (10,  35),
    "FOG":      (15,  45),
}


def _noise(sigma: float = 0.15) -> float:
    return random.gauss(0, sigma)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at fraction t (0–1)."""
    return a + (b - a) * t


def generate_history(
    days: int = 365,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate a list of daily driver readings for the past `days` days.

    Returns:
        List of dicts with keys:
            date, regime, growth, inflation, liquidity, volatility
    """
    random.seed(seed)
    records = []
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Build regime schedule backwards from today
    phases = _build_phases(days)

    day_cursor = today - timedelta(days=days - 1)

    for i in range(days):
        current_date = day_cursor + timedelta(days=i)
        phase, phase_day, phase_len, next_regime = _find_phase(phases, i)

        t_in_phase = phase_day / max(phase_len, 1)

        # Current and next regime driver centres
        c_cur  = REGIME_PATTERNS[phase]
        c_next = REGIME_PATTERNS[next_regime]

        # Smooth transition in final 20% of phase
        if t_in_phase > 0.80:
            blend = (t_in_phase - 0.80) / 0.20
        else:
            blend = 0.0

        g  = _lerp(c_cur[0], c_next[0], blend) + _noise(0.18)
        i  = _lerp(c_cur[1], c_next[1], blend) + _noise(0.15)
        lq = _lerp(c_cur[2], c_next[2], blend) + _noise(0.16)
        v  = _lerp(c_cur[3], c_next[3], blend) + _noise(0.20)

        records.append({
            "date":       current_date,
            "regime":     phase,
            "growth":     round(max(-2.0, min(2.0, g)),  3),
            "inflation":  round(max(-2.0, min(2.0, i)),  3),
            "liquidity":  round(max(-2.0, min(2.0, lq)), 3),
            "volatility": round(max(-2.0, min(2.0, v)),  3),
        })

    return records


def get_current_drivers(seed: int = 42) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return (current_drivers, previous_drivers) as raw score dicts.
    Uses the last two days of synthetic history.
    """
    history = generate_history(days=5, seed=seed)
    today     = history[-1]
    yesterday = history[-2]

    current = {
        "growth":    today["growth"],
        "inflation": today["inflation"],
        "liquidity": today["liquidity"],
        "volatility":today["volatility"],
    }
    previous = {
        "growth":    yesterday["growth"],
        "inflation": yesterday["inflation"],
        "liquidity": yesterday["liquidity"],
        "volatility":yesterday["volatility"],
    }
    return current, previous


def _build_phases(total_days: int) -> List[Tuple[str, int, int]]:
    """
    Build a list of (regime, start_day, duration) tuples covering total_days.
    """
    phases = []
    day = 0
    cycle_idx = 0
    random.seed(99)  # separate seed for phase lengths

    while day < total_days:
        regime = CYCLE[cycle_idx % len(CYCLE)]
        lo, hi = DURATION_RANGE[regime]
        dur = random.randint(lo, hi)
        dur = min(dur, total_days - day)
        phases.append((regime, day, dur))
        day += dur
        cycle_idx += 1

    return phases


def _find_phase(
    phases: List[Tuple[str, int, int]],
    day_index: int,
) -> Tuple[str, int, int, str]:
    """Return (regime, day_in_phase, phase_len, next_regime)."""
    for idx, (regime, start, dur) in enumerate(phases):
        end = start + dur
        if start <= day_index < end:
            next_idx = (idx + 1) % len(phases)
            next_regime = phases[next_idx][0]
            return regime, day_index - start, dur, next_regime
    # Fallback
    last = phases[-1]
    return last[0], 0, last[2], CYCLE[0]
