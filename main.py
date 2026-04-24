"""
Engine A — FastAPI Application
================================
Exposes the regime intelligence engine as a REST API.

Endpoints:
  GET  /                      Health check
  GET  /regime/summary        Full regime state + exposure + narrative + transitions
  GET  /regime/state          Current regime state only
  GET  /regime/drivers        Current driver readings
  GET  /regime/exposure       Exposure map for current regime
  GET  /regime/narrative      One-line narrative + risk flag
  GET  /regime/transitions    Transition probabilities
  GET  /regime/history        Historical regime data (query: ?days=90)
  POST /regime/simulate       Simulate regime with custom driver inputs
  GET  /regime/meta           Static regime metadata (colours, descriptions)

Deploy on Railway:
  - Add this service to your Railway project
  - Set PORT env var (Railway sets this automatically)
  - Optional: set USE_LIVE_DATA=true and connector API keys
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import List, Optional

from engine import EngineA, REGIME_META
from engine.models import (
    RegimeSummary, RegimeState, ExposureMap, Narrative,
    TransitionProb, HistoryPoint, SimulateRequest, RegimeCode
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GrineOS Engine A",
    description="Regime Intelligence API — SURGE · CRUISE · PRESSURE · SLIDE · SHOCK · FOG",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production to your GrineOS domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single engine instance (stateful cache inside)
_engine = EngineA()


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "service": "GrineOS Engine A",
        "version": "2.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Core regime endpoints ──────────────────────────────────────────────────────
@app.get("/regime/summary", response_model=RegimeSummary, tags=["Regime"])
def get_summary():
    """
    Full regime intelligence package:
    regime state + exposure map + narrative + transition probabilities.
    This is the primary endpoint for the GrineOS dashboard.
    """
    return _engine.get_summary()


@app.get("/regime/state", response_model=RegimeState, tags=["Regime"])
def get_state():
    """Current regime state: name, confidence, duration, drivers."""
    return _engine.get_summary().state


@app.get("/regime/drivers", tags=["Regime"])
def get_drivers():
    """Current driver readings: growth, inflation, liquidity, volatility."""
    state = _engine.get_summary().state
    return {
        "regime": state.code,
        "drivers": state.drivers,
        "timestamp": state.timestamp,
    }


@app.get("/regime/exposure", response_model=ExposureMap, tags=["Regime"])
def get_exposure():
    """Exposure map for the current regime — six asset classes with signals."""
    return _engine.get_summary().exposure


@app.get("/regime/narrative", response_model=Narrative, tags=["Regime"])
def get_narrative():
    """One-line narrative brief with horizon and risk flag."""
    return _engine.get_summary().narrative


@app.get("/regime/transitions", response_model=List[TransitionProb], tags=["Regime"])
def get_transitions():
    """30-day transition probabilities across all six regimes."""
    return _engine.get_summary().transitions


# ── History ────────────────────────────────────────────────────────────────────
@app.get("/regime/history", tags=["Regime"])
def get_history(days: int = Query(default=90, ge=1, le=365)):
    """
    Historical regime data for the dashboard timeline strip.
    Returns daily records: date, regime, confidence, driver scores.
    """
    raw_history = _engine.get_history(days)
    return {
        "days": len(raw_history),
        "history": [
            {
                "date":       r["date"].isoformat(),
                "regime":     r["regime"],
                "growth":     r["growth"],
                "inflation":  r["inflation"],
                "liquidity":  r["liquidity"],
                "volatility": r["volatility"],
            }
            for r in raw_history
        ],
    }


# ── Simulation ─────────────────────────────────────────────────────────────────
@app.post("/regime/simulate", response_model=RegimeSummary, tags=["Simulation"])
def simulate(req: SimulateRequest):
    """
    Simulate regime classification with custom driver inputs.
    Useful for scenario analysis and dashboard 'what-if' mode.

    Driver scores range from -2.0 (strongly negative) to +2.0 (strongly positive).
    """
    return _engine.simulate(
        growth=req.growth,
        inflation=req.inflation,
        liquidity=req.liquidity,
        volatility=req.volatility,
    )


# ── Static metadata ────────────────────────────────────────────────────────────
@app.get("/regime/meta", tags=["Metadata"])
def get_meta():
    """Static regime metadata: all six regimes with colours, descriptions, actions."""
    return {
        code: {
            "code":        meta.code.value,
            "subtitle":    meta.subtitle,
            "color":       meta.color,
            "tagline":     meta.tagline,
            "description": meta.description,
            "instinct":    meta.instinct,
        }
        for code, meta in REGIME_META.items()
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
