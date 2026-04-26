"""
Alert Engine
============
Monitors regime state and fires alerts when:
  1. Regime changes (e.g. SURGE -> CRUISE)
  2. Confidence drops below 80%
  3. Confidence delta > 5pts vs yesterday

State is stored in a simple JSON file on disk.
In production, swap for Redis or Postgres.

Alert channels:
  - Email via Resend (https://resend.com) - free tier: 100 emails/day
  - Easily extensible to Twilio SMS, Slack, etc.

Env vars required:
  RESEND_API_KEY    = your Resend API key
  ALERT_EMAIL_TO    = recipient email address
  ALERT_EMAIL_FROM  = sender address (e.g. alerts@yourdomain.com)
                      use onboarding@resend.dev for testing
"""

import os
import json
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


STATE_FILE = Path("/tmp/grineos_alert_state.json")

CONFIDENCE_THRESHOLD = 80.0   # alert when confidence drops below this
DELTA_THRESHOLD = 5.0         # alert when |delta| exceeds this


@dataclass
class AlertState:
    last_regime: str = "SURGE"
    last_confidence: float = 94.0
    last_check: str = ""
    alerts_sent: int = 0


def _load_state() -> AlertState:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text())
            return AlertState(**data)
        except Exception:
            pass
    return AlertState()


def _save_state(state: AlertState):
    STATE_FILE.write_text(json.dumps(asdict(state)))


def _send_email(subject: str, body_html: str) -> bool:
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("[Alerts] RESEND_API_KEY not set — email not sent")
        return False

    to_email = os.getenv("ALERT_EMAIL_TO", "kacy@grineos.com")
    from_email = os.getenv("ALERT_EMAIL_FROM", "onboarding@resend.dev")

    try:
        resp = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": f"GrineOS Alerts <{from_email}>",
                "to": [to_email],
                "subject": subject,
                "html": body_html,
            },
            timeout=10.0,
        )
        if resp.status_code in (200, 201):
            print(f"[Alerts] Email sent: {subject}")
            return True
        else:
            print(f"[Alerts] Email failed: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[Alerts] Email error: {e}")
        return False


def _regime_change_email(
    old_regime: str,
    new_regime: str,
    confidence: float,
    instinct: str,
    narrative: str,
    exposure_lines: str,
    regime_color: str,
) -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f8f9fb; margin: 0; padding: 0; }}
  .container {{ max-width: 600px; margin: 40px auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }}
  .header {{ background: #0f1923; padding: 32px 40px; }}
  .logo {{ font-family: Georgia, serif; font-size: 22px; font-weight: 700; color: #fff; letter-spacing: -0.02em; }}
  .badge {{ display: inline-block; margin-top: 12px; padding: 6px 14px; border-radius: 100px; background: {regime_color}22; border: 1px solid {regime_color}55; font-size: 11px; font-weight: 700; color: {regime_color}; letter-spacing: 0.1em; font-family: 'Courier New', monospace; }}
  .body {{ padding: 32px 40px; }}
  .change-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }}
  .regime-old {{ font-size: 18px; font-weight: 700; color: #94a3b8; font-family: Georgia, serif; }}
  .arrow {{ font-size: 20px; color: {regime_color}; font-weight: 700; }}
  .regime-new {{ font-size: 24px; font-weight: 700; color: {regime_color}; font-family: Georgia, serif; }}
  .conf {{ font-size: 13px; color: #64748b; margin-bottom: 20px; font-family: 'Courier New', monospace; }}
  .section {{ margin-bottom: 20px; }}
  .section-label {{ font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: {regime_color}; margin-bottom: 6px; font-family: 'Courier New', monospace; }}
  .section-body {{ font-size: 14px; line-height: 1.6; color: #1a2332; }}
  .exposure {{ background: #f8f9fb; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
  .exposure-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #e2e8f0; font-size: 13px; }}
  .exposure-row:last-child {{ border-bottom: none; }}
  .exp-asset {{ color: #475569; font-weight: 600; }}
  .exp-signal {{ font-family: 'Courier New', monospace; font-weight: 700; color: {regime_color}; }}
  .footer {{ background: #f1f5f9; padding: 20px 40px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
  .cta {{ display: inline-block; margin-top: 20px; padding: 12px 28px; background: #0f1923; color: #fff; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo">GrineOS</div>
    <div class="badge">REGIME CHANGE DETECTED</div>
  </div>
  <div class="body">
    <div class="change-row">
      <div class="regime-old">{old_regime}</div>
      <div class="arrow">&#8594;</div>
      <div class="regime-new">{new_regime}</div>
    </div>
    <div class="conf">Confidence: {confidence:.0f}%</div>
    <div class="section">
      <div class="section-label">Intelligence Brief</div>
      <div class="section-body">{narrative}</div>
    </div>
    <div class="section">
      <div class="section-label">Allocation Instinct</div>
      <div class="section-body">{instinct}</div>
    </div>
    <div class="section">
      <div class="section-label">Live Exposure Map</div>
      <div class="exposure">
        {''.join(f'<div class="exposure-row"><span class="exp-asset">{row.split(":")[0].strip()}</span><span class="exp-signal">{row.split(":")[1].strip() if ":" in row else ""}</span></div>' for row in exposure_lines.split(" | ") if row)}
      </div>
    </div>
    <a class="cta" href="https://kacygrine-cell.github.io/grineos-platform/">Open GrineOS Platform</a>
  </div>
  <div class="footer">
    GrineOS Engine A &middot; KG&Co Capital Advisory &middot; Grine Technologies Ltd<br/>
    This is an automated regime intelligence alert. Not investment advice.
  </div>
</div>
</body>
</html>
"""


def _confidence_alert_email(
    regime: str,
    confidence: float,
    delta: float,
    trigger: str,
    regime_color: str,
    risk_flag: str,
) -> str:
    trigger_desc = {
        "low_confidence": f"Confidence has dropped below {CONFIDENCE_THRESHOLD:.0f}% ({confidence:.0f}%)",
        "large_delta": f"Confidence moved {delta:+.1f} pts in 24 hours",
    }.get(trigger, trigger)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background: #f8f9fb; margin: 0; padding: 0; }}
  .container {{ max-width: 600px; margin: 40px auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }}
  .header {{ background: #0f1923; padding: 32px 40px; }}
  .logo {{ font-family: Georgia, serif; font-size: 22px; font-weight: 700; color: #fff; }}
  .badge {{ display: inline-block; margin-top: 12px; padding: 6px 14px; border-radius: 100px; background: #b4530922; border: 1px solid #b4530955; font-size: 11px; font-weight: 700; color: #b45309; letter-spacing: 0.1em; font-family: 'Courier New', monospace; }}
  .body {{ padding: 32px 40px; }}
  .regime {{ font-size: 28px; font-weight: 700; color: {regime_color}; font-family: Georgia, serif; margin-bottom: 8px; }}
  .trigger {{ font-size: 15px; color: #1a2332; margin-bottom: 20px; font-weight: 500; }}
  .conf-bar-wrap {{ background: #f1f5f9; border-radius: 6px; height: 8px; margin-bottom: 8px; overflow: hidden; }}
  .conf-bar {{ height: 100%; width: {min(confidence, 100):.0f}%; background: {'#16a34a' if confidence >= 80 else '#b45309' if confidence >= 60 else '#dc2626'}; border-radius: 6px; }}
  .conf-label {{ font-size: 12px; color: #64748b; margin-bottom: 20px; font-family: 'Courier New', monospace; }}
  .risk {{ background: #fffbeb; border: 1px solid #b4530933; border-radius: 8px; padding: 14px 18px; font-size: 13px; color: #1a2332; line-height: 1.6; }}
  .risk-label {{ font-size: 10px; font-weight: 700; color: #b45309; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; font-family: 'Courier New', monospace; }}
  .cta {{ display: inline-block; margin-top: 20px; padding: 12px 28px; background: #0f1923; color: #fff; text-decoration: none; border-radius: 6px; font-size: 13px; font-weight: 600; }}
  .footer {{ background: #f1f5f9; padding: 20px 40px; font-size: 12px; color: #94a3b8; border-top: 1px solid #e2e8f0; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="logo">GrineOS</div>
    <div class="badge">CONFIDENCE ALERT</div>
  </div>
  <div class="body">
    <div class="regime">{regime}</div>
    <div class="trigger">{trigger_desc}</div>
    <div class="conf-bar-wrap"><div class="conf-bar"></div></div>
    <div class="conf-label">Confidence: {confidence:.0f}% &nbsp; Delta: {delta:+.1f} pts vs yesterday</div>
    <div class="risk-label">Key Risk</div>
    <div class="risk">{risk_flag}</div>
    <a class="cta" href="https://kacygrine-cell.github.io/grineos-platform/">Open GrineOS Platform</a>
  </div>
  <div class="footer">
    GrineOS Engine A &middot; KG&Co Capital Advisory &middot; Grine Technologies Ltd<br/>
    This is an automated regime intelligence alert. Not investment advice.
  </div>
</div>
</body>
</html>
"""


def check_and_fire(summary) -> list:
    """
    Compare current regime state against saved state.
    Fire alerts for any triggered conditions.
    Returns list of alert types fired: e.g. ["regime_change", "low_confidence"]
    """
    state = _load_state()
    state_changed = False
    fired = []

    code = summary.state.code.value
    confidence = summary.state.confidence
    delta = summary.state.confidence_delta
    instinct = summary.state.instinct if hasattr(summary.state, 'instinct') else ""
    narrative = summary.narrative.text if summary.narrative else ""
    risk_flag = summary.narrative.risk_flag if summary.narrative else ""
    exposure_lines = " | ".join(
        f"{r.asset_class}: {r.signal}" for r in summary.exposure.rows
    ) if summary.exposure else ""

    from engine import REGIME_META
    meta = REGIME_META.get(code)
    regime_color = meta.color if meta else "#16a34a"
    instinct = meta.instinct if meta else instinct

    # 1. Regime change
    if state.last_regime and state.last_regime != code:
        subject = f"GrineOS: Regime Change — {state.last_regime} → {code}"
        body = _regime_change_email(
            old_regime=state.last_regime,
            new_regime=code,
            confidence=confidence,
            instinct=instinct,
            narrative=narrative,
            exposure_lines=exposure_lines,
            regime_color=regime_color,
        )
        if _send_email(subject, body):
            fired.append("regime_change")
            state.alerts_sent += 1
        state.last_regime = code
        state_changed = True

    # 2. Confidence below threshold
    if confidence < CONFIDENCE_THRESHOLD and state.last_confidence >= CONFIDENCE_THRESHOLD:
        subject = f"GrineOS: Low Confidence — {code} at {confidence:.0f}%"
        body = _confidence_alert_email(
            regime=code, confidence=confidence, delta=delta,
            trigger="low_confidence", regime_color=regime_color, risk_flag=risk_flag
        )
        if _send_email(subject, body):
            fired.append("low_confidence")
            state.alerts_sent += 1

    # 3. Large confidence delta
    if abs(delta) > DELTA_THRESHOLD:
        subject = f"GrineOS: Confidence Signal — {code} {delta:+.1f} pts in 24h"
        body = _confidence_alert_email(
            regime=code, confidence=confidence, delta=delta,
            trigger="large_delta", regime_color=regime_color, risk_flag=risk_flag
        )
        if _send_email(subject, body):
            fired.append("large_delta")
            state.alerts_sent += 1

    state.last_confidence = confidence
    state.last_check = datetime.now(timezone.utc).isoformat()

    if state_changed or fired:
        _save_state(state)

    return fired


def get_alert_status() -> dict:
    state = _load_state()
    return {
        "last_regime": state.last_regime,
        "last_confidence": state.last_confidence,
        "last_check": state.last_check,
        "alerts_sent": state.alerts_sent,
        "resend_configured": bool(os.getenv("RESEND_API_KEY")),
        "alert_email": os.getenv("ALERT_EMAIL_TO", "not set"),
        "thresholds": {
            "confidence_min": CONFIDENCE_THRESHOLD,
            "delta_max": DELTA_THRESHOLD,
        }
    }
