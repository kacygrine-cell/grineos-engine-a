"""
Engine A — Core Orchestrator
=============================
Wires together drivers → classifier → exposure → narrative
into a single RegimeSummary object.

Usage:
    from engine.core import EngineA
    engine = EngineA()
    summary = engine.get_summary()
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

from .models import RegimeCode, RegimeState, RegimeSummary, REGIME_META
from .drivers import score_drivers
from .classifier import classify
from .exposure import build_exposure
from .narrative import generate
from .data.synthetic import get_current_drivers, generate_history
from .data.connectors import get_live_drivers


USE_LIVE_DATA = os.getenv("USE_LIVE_DATA", "false").lower() == "true"


class EngineA:
    """
    Main engine class. Instantiate once, call get_summary() on each request.
    In production, add Redis/Postgres caching to avoid recomputing on every call.
    """

    def __init__(self):
        self._history_cache = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes

    # ── Public API ────────────────────────────────────────────────────────────
    def get_summary(self) -> RegimeSummary:
        """Return full regime summary: state + exposure + narrative + transitions."""
        raw_current, raw_previous = self._get_raw_drivers()
        drivers = score_drivers(raw_current, raw_previous)
        regime, confidence, transitions = classify(drivers)

        # Calculate duration and persistence from history
        duration_days, persistence_days, last_changed = self._calc_duration(regime)

        # Confidence delta vs yesterday
        yesterday_raw, day_before_raw = self._get_yesterday_drivers()
        yesterday_drivers = score_drivers(yesterday_raw, day_before_raw)
        _, yesterday_confidence, _ = classify(yesterday_drivers)
        confidence_delta = round(confidence - yesterday_confidence, 1)

        meta = REGIME_META[regime.value]
        state = RegimeState(
            code=regime,
            subtitle=meta.subtitle,
            color=meta.color,
            tagline=meta.tagline,
            description=meta.description,
            instinct=meta.instinct,
            confidence=round(confidence, 1),
            confidence_delta=confidence_delta,
            confidence_persistence=persistence_days,
            duration_days=duration_days,
            last_changed=last_changed,
            drivers=drivers,
            timestamp=datetime.now(timezone.utc),
        )

        exposure  = build_exposure(regime, drivers, confidence)
        narrative = generate(regime, drivers, confidence, confidence_delta, persistence_days)

        return RegimeSummary(
            state=state,
            exposure=exposure,
            narrative=narrative,
            transitions=transitions,
        )

    def get_history(self, days: int = 90) -> list:
        """Return historical regime data for the timeline strip."""
        return self._get_history(days)

    def simulate(
        self,
        growth: float,
        inflation: float,
        liquidity: float,
        volatility: float,
    ) -> RegimeSummary:
        """Simulate regime with custom driver inputs."""
        raw_current = {"growth": growth, "inflation": inflation,
                       "liquidity": liquidity, "volatility": volatility}
        raw_previous = raw_current  # No delta in simulation
        drivers = score_drivers(raw_current, raw_previous)
        regime, confidence, transitions = classify(drivers)
        meta = REGIME_META[regime.value]

        state = RegimeState(
            code=regime,
            subtitle=meta.subtitle,
            color=meta.color,
            tagline=meta.tagline,
            description=meta.description,
            instinct=meta.instinct,
            confidence=round(confidence, 1),
            confidence_delta=0.0,
            confidence_persistence=0,
            duration_days=0,
            last_changed=datetime.now(timezone.utc),
            drivers=drivers,
            timestamp=datetime.now(timezone.utc),
        )
        exposure  = build_exposure(regime, drivers, confidence)
        narrative = generate(regime, drivers, confidence, 0.0, 0)

        return RegimeSummary(
            state=state,
            exposure=exposure,
            narrative=narrative,
            transitions=transitions,
        )

    # ── Private helpers ───────────────────────────────────────────────────────
    def _get_raw_drivers(self):
        if USE_LIVE_DATA:
            live = get_live_drivers()
            if live:
                return live
        return get_current_drivers()

    def _get_yesterday_drivers(self):
        from .data.synthetic import generate_history
        h = generate_history(days=3)
        d1 = h[-2]
        d2 = h[-3]
        cur  = {"growth": d1["growth"], "inflation": d1["inflation"],
                "liquidity": d1["liquidity"], "volatility": d1["volatility"]}
        prev = {"growth": d2["growth"], "inflation": d2["inflation"],
                "liquidity": d2["liquidity"], "volatility": d2["volatility"]}
        return cur, prev

    def _get_history(self, days: int = 90):
        now = datetime.now(timezone.utc)
        if (self._history_cache is None or self._cache_time is None or
                (now - self._cache_time).total_seconds() > self._cache_ttl_seconds):
            self._history_cache = generate_history(days=max(days + 10, 100))
            self._cache_time = now
        return self._history_cache[-days:]

    def _calc_duration(self, current_regime: RegimeCode):
        history = self._get_history(180)
        # Walk backwards from today to find consecutive days in this regime
        duration = 0
        persistence = 0
        last_changed = datetime.now(timezone.utc)
        found_change = False

        for record in reversed(history):
            if record["regime"] == current_regime.value:
                duration += 1
            else:
                if not found_change:
                    last_changed = record["date"] + timedelta(days=1)
                    found_change = True
                break

        # Persistence: days confidence would be above 80 (approximate from history)
        for record in reversed(history):
            if record["regime"] == current_regime.value:
                persistence += 1
            else:
                break

        if not found_change:
            last_changed = (datetime.now(timezone.utc) - timedelta(days=duration))

        return max(duration, 1), max(persistence, 1), last_changed
