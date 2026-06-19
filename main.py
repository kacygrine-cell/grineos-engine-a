import os
import logging
logger = logging.getLogger(__name__)
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

from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

_engine = EngineA()

# Check if live data mode is enabled
USE_LIVE = os.getenv("USE_LIVE_DATA", "false").lower() == "true"

def _get_live_summary():
    """Get regime summary using real market data drivers."""
    from engine.live_drivers import fetch_live_drivers
    scores = fetch_live_drivers()
    return _engine.simulate(
        growth=scores["growth"],
        inflation=scores["inflation"],
        liquidity=scores["liquidity"],
        volatility=scores["volatility"],
    )

def _get_summary():
    """Return live or synthetic summary based on USE_LIVE_DATA env var."""
    if USE_LIVE:
        try:
            return _get_live_summary()
        except Exception as e:
            logger.warning(f"Live data failed, falling back to synthetic: {e}")
    return _get_summary()


#  Health 
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "service": "GrineOS Engine A",
        "version": "2.0.0",
        "live_data_mode": USE_LIVE,
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/regime/drivers/live", tags=["Regime"])
def get_live_drivers():
    """Fetch and return real-time driver scores from Yahoo Finance."""
    from engine.live_drivers import fetch_live_drivers, get_driver_labels
    try:
        scores = fetch_live_drivers()
        labels = get_driver_labels(scores)
        return {
            "source": "yahoo_finance",
            "scores": scores,
            "labels": labels,
            "live_mode_active": USE_LIVE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live driver fetch failed: {str(e)}")


#  Regime endpoints 
@app.get("/regime/summary", response_model=RegimeSummary, tags=["Regime"])
def get_summary():
    return _get_summary()

@app.get("/regime/state", response_model=RegimeState, tags=["Regime"])
def get_state():
    return _get_summary().state

@app.get("/regime/drivers", tags=["Regime"])
def get_drivers():
    state = _get_summary().state
    return {"regime": state.code, "drivers": state.drivers, "timestamp": state.timestamp}

@app.get("/regime/exposure", response_model=ExposureMap, tags=["Regime"])
def get_exposure():
    return _get_summary().exposure

@app.get("/regime/narrative", response_model=Narrative, tags=["Regime"])
def get_narrative():
    return _get_summary().narrative

@app.get("/regime/transitions", response_model=List[TransitionProb], tags=["Regime"])
def get_transitions():
    return _get_summary().transitions

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

    summary = _get_summary()
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
        "You are Grine, an institutional investment intelligence system built by KG&Co Capital Advisory. "
        "You combine deep macro expertise with the live regime data below. "
        "You think like a seasoned CIO who has managed capital through multiple cycles - direct, intellectually honest, and genuinely curious.\n\n"
        "Your character:\n"
        "- You adapt your response to what's actually being asked. Simple question = direct answer. Complex question = structured analysis.\n"
        "- You have genuine opinions and share them confidently, but acknowledge uncertainty when it exists.\n"
        "- You are not a yes machine. If someone's portfolio positioning looks wrong given the current regime, you say so.\n"
        "- You ask a clarifying question when the answer genuinely depends on something you don't know.\n"
        "- You use numbers and specifics, not vague language.\n"
        "- You are warm and direct, never cold or bureaucratic.\n"
        "- You never repeat the same four-section template regardless of what was asked.\n\n"
        "LIVE MARKET STATE (Engine A - real Yahoo Finance data):\n"
        f"Regime: {state.code.value} - {state.subtitle}\n"
        f"Confidence: {state.confidence:.0f}% ({state.confidence_delta:+.1f} pts vs yesterday)\n"
        f"Duration: {state.duration_days} days in this regime\n"
        f"Narrative: {narrative_text}\n"
        f"Growth: {g.label if g else 'n/a'} ({g.score:+.2f}s)\n"
        f"Inflation: {i.label if i else 'n/a'} ({i.score:+.2f}s)\n"
        f"Liquidity: {l.label if l else 'n/a'} ({l.score:+.2f}s)\n"
        f"Volatility: {v.label if v else 'n/a'} ({v.score:+.2f}s)\n"
        f"Allocation instinct: {REGIME_META[state.code.value].instinct}\n"
        f"Exposure map: {exposure_lines}\n"
        f"Key risk: {risk_flag}\n"
        + (f"Portfolio context: {req.portfolio_summary}\n" if req.portfolio_summary else "")
        + "\nUse this live data to ground every response in current market reality. "
        "Reference specific driver scores and regime signals when they're relevant to what's being asked. "
        "Format your response appropriately for the question - sometimes a single sentence is the right answer, "
        "sometimes a structured breakdown is what's needed. Use **bold** for emphasis on key points. "
        "Never start with 'Great question' or similar filler. Get straight to the point."
    )

    messages = [{"role": m.role, "content": m.content} for m in req.history[-6:]]
    messages.append({"role": "user", "content": req.message})

    # Web search tool definition
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
        }
    ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Agentic loop: keep calling until no more tool use
        current_messages = messages.copy()
        max_iterations = 5
        final_reply = "No response."

        for _ in range(max_iterations):
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1500,
                    "system": system_prompt,
                    "tools": tools,
                    "messages": current_messages,
                }
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Anthropic error {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            stop_reason = data.get("stop_reason")
            content_blocks = data.get("content", [])

            # Extract any text from this response
            text_parts = [b["text"] for b in content_blocks if b.get("type") == "text" and b.get("text")]
            if text_parts:
                final_reply = " ".join(text_parts)

            # If done, break
            if stop_reason == "end_turn":
                break

            # If tool use, add assistant message and continue
            if stop_reason == "tool_use":
                current_messages.append({"role": "assistant", "content": content_blocks})
                # Add tool results (web_search handles results automatically in the API)
                tool_results = []
                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": "Search completed."
                        })
                if tool_results:
                    current_messages.append({"role": "user", "content": tool_results})
                continue

            break

    return AgentResponse(response=final_reply, regime=state.code.value, confidence=state.confidence)


#  Alerts 
from engine.alert import check_and_fire, get_alert_status

class AlertConfigRequest(BaseModel):
    email: str
    active: bool = True

@app.post("/alerts/check", tags=["Alerts"])
async def alerts_check():
    summary = _get_summary()
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
    summary = _get_summary()
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
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@app.get("/backtest/test", tags=["Backtest"])
def backtest_test():
    """Diagnostic: verify yfinance and Yahoo Finance connectivity."""
    results = {}
    try:
        import yfinance as yf
        results["yfinance_import"] = "OK"
    except ImportError as e:
        results["yfinance_import"] = f"FAILED: {e}"
        return results
    try:
        import pandas as pd
        results["pandas_import"] = "OK"
    except ImportError as e:
        results["pandas_import"] = f"FAILED: {e}"
    try:
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="5d")
        if len(hist) > 0:
            results["yahoo_finance"] = f"OK - {len(hist)} days of SPY data"
            results["latest_spy"] = float(hist["Close"].iloc[-1])
        else:
            results["yahoo_finance"] = "FAILED - empty response"
    except Exception as e:
        results["yahoo_finance"] = f"FAILED: {type(e).__name__}: {str(e)}"
    return results

@app.post("/backtest/run", tags=["Backtest"])
def backtest_run(req: BacktestRequest):
    period_map = {"1y": 1, "2y": 2, "3y": 3, "5y": 5, "7y": 7, "10y": 10, "max": 15}
    period_years = period_map.get(req.period, 3)
    # "max" = from 2010-01-01 to today
    start_override = req.start_date
    end_override = req.end_date
    if req.period == "max" and not start_override:
        start_override = "2010-01-01"
    holdings = [{"ticker": h.ticker, "weight": h.weight} for h in req.holdings]
    try:
        result = run_backtest(
            holdings=holdings,
            period_years=period_years,
            rebalance_frequency=req.rebalance,
            profile_multiplier=req.profile_multiplier,
            start_date_override=start_override,
            end_date_override=end_override,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {type(e).__name__}: {str(e)}")


#  Regime Replay 
@app.get("/regime/replay", tags=["Replay"])
def regime_replay(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """Reconstruct what the regime would have been on any historical date."""
    from datetime import datetime, timedelta
    try:
        target_dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    if target_dt < datetime(2010, 1, 1):
        raise HTTPException(status_code=400, detail="Date must be 2010-01-01 or later.")
    if target_dt > datetime.today():
        raise HTTPException(status_code=400, detail="Date cannot be in the future.")
    try:
        import yfinance as yf
        import pandas as pd
        import math

        start = (target_dt - timedelta(days=400)).strftime("%Y-%m-%d")
        end = (target_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        tickers = ["SPY", "^VIX", "TIP", "IEF", "HYG", "LQD"]
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, :6]
        else:
            prices = raw
        prices = prices.ffill().dropna(how="all")
        prices = prices[prices.index <= pd.Timestamp(target_dt)]
        if prices.empty or len(prices) < 20:
            raise ValueError("Insufficient data for this date.")
        actual_date = prices.index[-1]

        def get_col(name):
            clean = name.replace("^", "")
            for c in [name, clean, name.upper()]:
                if c in prices.columns:
                    return prices[c].dropna()
            return pd.Series(dtype=float)

        def zscore_clip(series, value, clip=2.0):
            vals = series.tolist()
            if len(vals) < 10: return 0.0
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v-mean)**2 for v in vals)/len(vals)) or 1.0
            return max(-clip, min(clip, (value-mean)/std))

        scores = {"growth": 0.0, "inflation": 0.0, "liquidity": 0.0, "volatility": 0.0}
        spy = get_col("SPY")
        if len(spy) >= 64:
            mom = spy.pct_change(63).dropna()
            if len(mom) >= 2: scores["growth"] = round(zscore_clip(mom, float(mom.iloc[-1])), 3)
        vix = get_col("^VIX")
        if len(vix) >= 20: scores["volatility"] = round(-zscore_clip(vix, float(vix.iloc[-1])), 3)
        tip, ief = get_col("TIP"), get_col("IEF")
        if len(tip) >= 64 and len(ief) >= 64:
            aln = pd.concat([tip, ief], axis=1, join="inner").dropna()
            if len(aln) >= 64:
                inf_mom = (aln.iloc[:,0]/aln.iloc[:,1]).pct_change(63).dropna()
                if len(inf_mom) >= 2: scores["inflation"] = round(zscore_clip(inf_mom, float(inf_mom.iloc[-1])), 3)
        hyg, lqd = get_col("HYG"), get_col("LQD")
        if len(hyg) >= 20 and len(lqd) >= 20:
            aln2 = pd.concat([hyg, lqd], axis=1, join="inner").dropna()
            if len(aln2) >= 20:
                cr = aln2.iloc[:,0]/aln2.iloc[:,1]
                scores["liquidity"] = round(zscore_clip(cr, float(cr.iloc[-1])), 3)

        summary = _engine.simulate(
            growth=scores["growth"], inflation=scores["inflation"],
            liquidity=scores["liquidity"], volatility=scores["volatility"],
        )
        meta = REGIME_META.get(summary.state.code.value)
        lbl = {"growth": "Accelerating" if scores["growth"]>0.5 else "Decelerating" if scores["growth"]<-0.5 else "Stable",
               "inflation": "Rising" if scores["inflation"]>0.5 else "Falling" if scores["inflation"]<-0.5 else "Contained",
               "liquidity": "Abundant" if scores["liquidity"]>0.5 else "Tightening" if scores["liquidity"]<-0.5 else "Neutral",
               "volatility": "Suppressed" if scores["volatility"]>0.5 else "Elevated" if scores["volatility"]<-0.5 else "Moderate"}
        return {
            "requested_date": date, "actual_date": str(actual_date.date()),
            "regime": summary.state.code.value, "subtitle": summary.state.subtitle,
            "color": meta.color if meta else "#16a34a",
            "confidence": round(summary.state.confidence, 1),
            "instinct": meta.instinct if meta else "",
            "drivers": {k: {"score": scores[k], "label": lbl[k]} for k in scores},
            "narrative": summary.narrative.text if summary.narrative else "",
            "risk_flag": summary.narrative.risk_flag if summary.narrative else "",
            "exposure_rows": [{"asset_class": r.asset_class, "signal": r.signal, "direction": r.direction,
                "magnitude": r.magnitude, "confidence": r.confidence} for r in summary.exposure.rows] if summary.exposure else [],
            "transitions": [{"regime": t.regime, "probability": t.probability} for t in summary.transitions] if summary.transitions else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Replay error: {type(e).__name__}: {str(e)}")

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

    summary = _get_summary()
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
            json={"model": "claude-sonnet-4-6", "max_tokens": 2000, "system": system_prompt,
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
