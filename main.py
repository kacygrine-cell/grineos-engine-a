import os
import json as _json
import re as _re
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


#  Health 
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "service": "GrineOS Engine A",
        "version": "2.0.0",
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


#  Regime endpoints 
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


#  Agent 
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
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    summary = _engine.get_summary()
    state = summary.state
    exposure_lines = " | ".join(
        f"{r.asset_class}: {r.signal}" for r in summary.exposure.rows
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
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 800, "system": system_prompt, "messages": messages}
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Anthropic error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    reply = data["content"][0]["text"] if data.get("content") else "No response."
    return AgentResponse(response=reply, regime=state.code.value, confidence=state.confidence)


#  Alerts 
from engine.alert import check_and_fire, get_alert_status

class AlertConfigRequest(BaseModel):
    email: str
    active: bool = True

@app.post("/alerts/check", tags=["Alerts"])
async def alerts_check():
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
    return get_alert_status()

@app.post("/alerts/test", tags=["Alerts"])
async def alerts_test():
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="RESEND_API_KEY not configured.")
    to_email = os.getenv("ALERT_EMAIL_TO", "")
    if not to_email:
        raise HTTPException(status_code=503, detail="ALERT_EMAIL_TO not configured.")
    summary = _engine.get_summary()
    state = summary.state
    meta = REGIME_META.get(state.code.value)
    html = (
        '<div style="font-family:Arial,sans-serif;max-width:500px;margin:40px auto;padding:32px;'
        'background:#fff;border-radius:12px;border:1px solid #e2e8f0">'
        '<div style="font-family:Georgia,serif;font-size:20px;font-weight:700;color:#0f1923;margin-bottom:16px">'
        'GrineOS Test Alert</div>'
        '<p style="color:#475569;margin-bottom:16px">Your alert system is configured correctly.</p>'
        '<div style="background:#f8f9fb;border-radius:8px;padding:16px;font-family:Courier New,monospace;font-size:13px;color:#1a2332">'
        f'<div>Regime: <strong style="color:{meta.color if meta else "#16a34a"}">{state.code.value}</strong></div>'
        f'<div>Confidence: {state.confidence:.0f}%</div>'
        f'<div>Delta: {state.confidence_delta:+.1f} pts</div>'
        '</div></div>'
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
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


#  Backtest 
from engine.backtest import run_backtest

class BacktestHolding(BaseModel):
    ticker: str
    weight: float

class BacktestRequest(BaseModel):
    holdings: List[BacktestHolding]
    period: str = "3y"
    rebalance: str = "monthly"
    profile_multiplier: float = 1.0

@app.post("/backtest/run", tags=["Backtest"])
async def backtest_run(req: BacktestRequest):
    period_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}
    period_years = period_map.get(req.period, 3)
    holdings = [{"ticker": h.ticker, "weight": h.weight} for h in req.holdings]
    result = run_backtest(
        holdings=holdings,
        period_years=period_years,
        rebalance_frequency=req.rebalance,
        profile_multiplier=req.profile_multiplier,
    )
    return result


#  Investment Committee Brief 
class ICBriefRequest(BaseModel):
    portfolio_summary: Optional[str] = None
    client_name: Optional[str] = "Investment Committee"
    since_days: int = 30

class ICBriefResponse(BaseModel):
    regime: str
    confidence: float
    subtitle: str
    color: str
    generated_at: str
    content: dict

@app.post("/ic/brief", tags=["IC Mode"])
async def ic_brief(req: ICBriefRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    summary = _engine.get_summary()
    state = summary.state
    meta = REGIME_META.get(state.code.value)
    g = state.drivers.get("growth")
    i = state.drivers.get("inflation")
    l = state.drivers.get("liquidity")
    v = state.drivers.get("volatility")

    exposure_text = "\n".join(
        f"- {r.asset_class}: {r.signal} (confidence {r.confidence:.0f}%)"
        for r in summary.exposure.rows
    ) if summary.exposure else "Not available"

    transitions_text = "\n".join(
        f"- {t.regime}: {t.probability * 100:.0f}%"
        for t in summary.transitions
    ) if summary.transitions else ""

    system_prompt = """You are a senior CIO preparing a formal Investment Committee brief.
Your output must be a JSON object with exactly these keys (no markdown fences):

{
  "executive_summary": "2-3 sentence overview of current regime and key implications",
  "what_changed": ["bullet 1", "bullet 2", "bullet 3"],
  "top_recommendations": [
    {"action": "action text", "rationale": "one sentence why", "urgency": "immediate|this week|this month"},
    {"action": "action text", "rationale": "one sentence why", "urgency": "immediate|this week|this month"},
    {"action": "action text", "rationale": "one sentence why", "urgency": "immediate|this week|this month"},
    {"action": "action text", "rationale": "one sentence why", "urgency": "immediate|this week|this month"},
    {"action": "action text", "rationale": "one sentence why", "urgency": "immediate|this week|this month"}
  ],
  "historical_analogue": {"period": "e.g. 2017 Q2", "description": "2 sentences on similarity and what happened", "outcome": "what the right trade was"},
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "alternative_scenario": {"trigger": "what would cause regime change", "new_regime": "regime name", "portfolio_action": "what to do immediately"}
}

Be specific, institutional, and decisive. No hedging. Every recommendation must be actionable."""

    user_prompt = (
        f"Generate an Investment Committee brief for {req.client_name}.\n\n"
        f"LIVE REGIME STATE:\n"
        f"Regime: {state.code.value} - {state.subtitle}\n"
        f"Confidence: {state.confidence:.0f}% ({state.confidence_delta:+.1f} pts vs yesterday)\n"
        f"Duration: {state.duration_days} days in this regime\n"
        f"Narrative: {summary.narrative.text if summary.narrative else 'n/a'}\n\n"
        f"DRIVER READINGS:\n"
        f"Growth: {g.label if g else 'n/a'} ({g.score:+.2f}s)\n"
        f"Inflation: {i.label if i else 'n/a'} ({i.score:+.2f}s)\n"
        f"Liquidity: {l.label if l else 'n/a'} ({l.score:+.2f}s)\n"
        f"Volatility: {v.label if v else 'n/a'} ({v.score:+.2f}s)\n\n"
        f"ALLOCATION INSTINCT:\n{meta.instinct if meta else 'n/a'}\n\n"
        f"LIVE EXPOSURE MAP:\n{exposure_text}\n\n"
        f"30D TRANSITION PROBABILITIES:\n{transitions_text}\n\n"
        f"KEY RISK: {summary.narrative.risk_flag if summary.narrative else 'n/a'}\n\n"
        f"Portfolio context: {req.portfolio_summary or 'Institutional multi-asset portfolio'}"
    )

    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-sonnet-4-20250514", "max_tokens": 2000, "system": system_prompt,
                  "messages": [{"role": "user", "content": user_prompt}]}
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Anthropic error: {resp.text[:200]}")

    raw = resp.json()["content"][0]["text"]
    clean = _re.sub(r"```(?:json)?|```", "", raw).strip()
    content = _json.loads(clean)

    exposure_rows = [
        {"asset_class": r.asset_class, "signal": r.signal,
         "direction": r.direction, "confidence": r.confidence}
        for r in summary.exposure.rows
    ] if summary.exposure else []

    transitions_list = [
        {"regime": t.regime, "probability": t.probability}
        for t in summary.transitions
    ] if summary.transitions else []

    return ICBriefResponse(
        regime=state.code.value,
        confidence=state.confidence,
        subtitle=state.subtitle,
        color=meta.color if meta else "#16a34a",
        generated_at=datetime.now(timezone.utc).isoformat(),
        content={
            **content,
            "regime": state.code.value,
            "subtitle": state.subtitle,
            "confidence": state.confidence,
            "confidence_delta": state.confidence_delta,
            "duration_days": state.duration_days,
            "narrative": summary.narrative.text if summary.narrative else "",
            "risk_flag": summary.narrative.risk_flag if summary.narrative else "",
            "instinct": meta.instinct if meta else "",
            "drivers": {
                "growth":     {"label": g.label, "score": g.score} if g else {},
                "inflation":  {"label": i.label, "score": i.score} if i else {},
                "liquidity":  {"label": l.label, "score": l.score} if l else {},
                "volatility": {"label": v.label, "score": v.score} if v else {},
            },
            "exposure_rows": exposure_rows,
            "transitions":   transitions_list,
            "color": meta.color if meta else "#16a34a",
        }
    )


#  Entry point 
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
