import os
import httpx
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel

from engine import EngineA, REGIME_META
from engine.models import (
    RegimeSummary, RegimeState, ExposureMap, Narrative,
    TransitionProb, SimulateRequest, RegimeCode
)

app = FastAPI(
    title="GrineOS Engine A",
    description="Regime Intelligence API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = EngineA()


@app.get("/", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "service": "GrineOS Engine A",
        "version": "2.0.0",
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/regime/summary", response_model=RegimeSummary, tags=["Regime"])
def get_summary():
    return _engine.get_summary()


@app.get("/regime/state", response_model=RegimeState, tags=["Regime"])
def get_state():
    return _engine.get_summary().state


@app.get("/regime/drivers", tags=["Regime"])
def get_drivers():
    state = _engine.get_summary().state
    return {"regime": state.code, "drivers": state.drivers, "timestamp": state.timestamp}


@app.get("/regime/exposure", response_model=ExposureMap, tags=["Regime"])
def get_exposure():
    return _engine.get_summary().exposure


@app.get("/regime/narrative", response_model=Narrative, tags=["Regime"])
def get_narrative():
    return _engine.get_summary().narrative


@app.get("/regime/transitions", response_model=List[TransitionProb], tags=["Regime"])
def get_transitions():
    return _engine.get_summary().transitions


@app.get("/regime/history", tags=["Regime"])
def get_history(days: int = Query(default=90, ge=1, le=365)):
    raw = _engine.get_history(days)
    return {
        "days": len(raw),
        "history": [
            {
                "date":       r["date"].isoformat(),
                "regime":     r["regime"],
                "growth":     r["growth"],
                "inflation":  r["inflation"],
                "liquidity":  r["liquidity"],
                "volatility": r["volatility"],
            }
            for r in raw
        ],
    }


@app.post("/regime/simulate", response_model=RegimeSummary, tags=["Simulation"])
def simulate(req: SimulateRequest):
    return _engine.simulate(
        growth=req.growth,
        inflation=req.inflation,
        liquidity=req.liquidity,
        volatility=req.volatility,
    )


@app.get("/regime/meta", tags=["Metadata"])
def get_meta():
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


class AgentMessage(BaseModel):
    role: str
    content: str

class AgentRequest(BaseModel):
    message: str
    history: List[AgentMessage] = []
    portfolio_summary: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    regime: str
    confidence: float


@app.post("/agent/chat", response_model=AgentResponse, tags=["Agent"])
async def agent_chat(req: AgentRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured on this server."
        )

    summary = _engine.get_summary()
    state = summary.state

    exposure_lines = " | ".join(
        f"{r.asset_class}: {r.signal}"
        for r in summary.exposure.rows
    ) if summary.exposure else "not available"

    narrative_text = summary.narrative.text if summary.narrative else "not available"
    risk_flag = summary.narrative.risk_flag if summary.narrative else "not available"

    g = state.drivers.get("growth")
    i = state.drivers.get("inflation")
    l = state.drivers.get("liquidity")
    v = state.drivers.get("volatility")

    system_prompt = (
        "You are a senior CIO at an institutional asset manager. Direct and decisive.\n\n"
        "LIVE MARKET STATE (GrineOS Engine A):\n"
        f"Regime: {state.code.value} - {state.subtitle}\n"
        f"Confidence: {state.confidence:.0f}% ({state.confidence_delta:+.1f} pts vs yesterday)\n"
        f"Duration: {state.duration_days} days\n"
        f"Narrative: {narrative_text}\n"
        f"Growth: {g.label if g else 'n/a'} ({g.score:+.2f}s)\n"
        f"Inflation: {i.label if i else 'n/a'} ({i.score:+.2f}s)\n"
        f"Liquidity: {l.label if l else 'n/a'} ({l.score:+.2f}s)\n"
        f"Volatility: {v.label if v else 'n/a'} ({v.score:+.2f}s)\n"
        f"Instinct: {REGIME_META[state.code.value].instinct}\n"
        f"Exposure: {exposure_lines}\n"
        f"Key risk: {risk_flag}\n"
        + (f"Portfolio: {req.portfolio_summary}\n" if req.portfolio_summary else "")
        + "\nRESPONSE FORMAT:\n"
        "**REGIME ASSESSMENT:** One sentence.\n"
        "**MARKET VIEW:** What drivers say now.\n"
        "**ALLOCATION GUIDANCE:** Specific and actionable.\n"
        "**KEY RISK:** One thing that could change everything.\n\n"
        "Be concise. Be decisive. No hedging."
    )

    messages = [{"role": m.role, "content": m.content} for m in req.history[-6:]]
    messages.append({"role": "user", "content": req.message})

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 800,
                "system": system_prompt,
                "messages": messages,
            }
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Anthropic API error {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    reply = data["content"][0]["text"] if data.get("content") else "No response."

    return AgentResponse(
        response=reply,
        regime=state.code.value,
        confidence=state.confidence,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
