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


# Alert endpoints
from engine.alerts import check_and_fire, get_alert_status

class AlertConfigRequest(BaseModel):
    email: str
    active: bool = True

@app.post("/alerts/check", tags=["Alerts"])
async def alerts_check():
    """
    Check current regime state and fire any triggered alerts.
    Call this on a schedule (e.g. Railway Cron: every 30 minutes).
    """
    summary = _engine.get_summary()
    fired = check_and_fire(summary)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "regime": summary.state.code.value,
        "confidence": summary.state.confidence,
        "confidence_delta": summary.state.confidence_delta,
        "alerts_fired": fired,
        "alerts_count": len(fired),
    }

@app.get("/alerts/status", tags=["Alerts"])
def alerts_status():
    """Current alert system status and configuration."""
    return get_alert_status()

@app.post("/alerts/test", tags=["Alerts"])
async def alerts_test():
    """
    Send a test alert email to verify configuration.
    Requires RESEND_API_KEY and ALERT_EMAIL_TO to be set.
    """
    import os, httpx as _httpx
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="RESEND_API_KEY not configured.")
    to_email = os.getenv("ALERT_EMAIL_TO", "")
    if not to_email:
        raise HTTPException(status_code=503, detail="ALERT_EMAIL_TO not configured.")

    summary = _engine.get_summary()
    state = summary.state
    from engine import REGIME_META
    meta = REGIME_META.get(state.code.value)

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:500px;margin:40px auto;padding:32px;background:#fff;border-radius:12px;border:1px solid #e2e8f0">
      <div style="font-family:Georgia,serif;font-size:20px;font-weight:700;color:#0f1923;margin-bottom:16px">GrineOS Test Alert</div>
      <p style="color:#475569;margin-bottom:16px">Your alert system is configured correctly.</p>
      <div style="background:#f8f9fb;border-radius:8px;padding:16px;font-family:'Courier New',monospace;font-size:13px;color:#1a2332">
        <div>Current Regime: <strong style="color:{meta.color if meta else '#16a34a'}">{state.code.value}</strong></div>
        <div>Confidence: {state.confidence:.0f}%</div>
        <div>Delta: {state.confidence_delta:+.1f} pts</div>
      </div>
      <p style="color:#94a3b8;font-size:12px;margin-top:16px">GrineOS Engine A &middot; KG&Co Capital Advisory</p>
    </div>
    """
    async with _httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from": f"GrineOS Alerts <{os.getenv('ALERT_EMAIL_FROM', 'onboarding@resend.dev')}>",
                "to": [to_email],
                "subject": f"GrineOS Test Alert - {state.code.value} {state.confidence:.0f}%",
                "html": html,
            }
        )
    if resp.status_code in (200, 201):
        return {"sent": True, "to": to_email, "regime": state.code.value}
    raise HTTPException(status_code=502, detail=f"Resend error: {resp.text[:200]}")
