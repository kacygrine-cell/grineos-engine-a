from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime


class RegimeCode(str, Enum):
    SURGE    = "SURGE"
    CRUISE   = "CRUISE"
    PRESSURE = "PRESSURE"
    SLIDE    = "SLIDE"
    SHOCK    = "SHOCK"
    FOG      = "FOG"


class RegimeMeta(BaseModel):
    code: RegimeCode
    subtitle: str
    color: str
    tagline: str
    description: str
    instinct: str


REGIME_META: Dict[str, RegimeMeta] = {
    "SURGE": RegimeMeta(
        code=RegimeCode.SURGE, subtitle="Risk Expansion", color="#16a34a",
        tagline="All systems go. Full risk deployment.",
        description="Growth accelerating, inflation contained, liquidity abundant, volatility suppressed.",
        instinct="Deploy maximum risk. Overweight equities, credit, commodities. Reduce cash to floor.",
    ),
    "CRUISE": RegimeMeta(
        code=RegimeCode.CRUISE, subtitle="Cycle Momentum", color="#22c55e",
        tagline="Mature momentum. Selective risk.",
        description="Bull market in full swing but leadership narrowing. Growth positive, inflation rising.",
        instinct="Stay invested. Favour quality equities. Begin trimming duration.",
    ),
    "PRESSURE": RegimeMeta(
        code=RegimeCode.PRESSURE, subtitle="Inflation Squeeze", color="#b45309",
        tagline="Growth fading. Prices holding. Protect purchasing power.",
        description="Growth decelerating while inflation stays elevated. 60/40 structurally challenged.",
        instinct="Cut duration and core equities. Overweight commodities and inflation hedges.",
    ),
    "SLIDE": RegimeMeta(
        code=RegimeCode.SLIDE, subtitle="Growth Breakdown", color="#f97316",
        tagline="Cycle turning. Rotate to defence.",
        description="Growth in recession territory. Earnings deteriorating. Credit spreads widening.",
        instinct="De-risk immediately. Overweight defensives and quality bonds. Exit high beta.",
    ),
    "SHOCK": RegimeMeta(
        code=RegimeCode.SHOCK, subtitle="Liquidity Break", color="#dc2626",
        tagline="Capital at risk. Preservation is the only priority.",
        description="Liquidity seizes. Volatility spikes. Correlations collapse to 1.",
        instinct="Maximum defensive. Cash and sovereign bonds only. Review every position.",
    ),
    "FOG": RegimeMeta(
        code=RegimeCode.FOG, subtitle="Regime Uncertainty", color="#7c3aed",
        tagline="Signals disagree. Wait for clarity.",
        description="Mixed drivers. High model uncertainty. Dangerous moment for high-conviction bets.",
        instinct="Balanced positioning. Reduce conviction sizing. Wait for confirmation.",
    ),
}


class DriverScore(BaseModel):
    name: str
    score: float = Field(..., ge=-2.0, le=2.0)
    label: str
    confidence: float = Field(..., ge=0.0, le=100.0)
    trend: str  # "rising" | "falling" | "stable"
    raw_value: Optional[float] = None
    raw_label: Optional[str] = None


class RegimeState(BaseModel):
    code: RegimeCode
    subtitle: str
    color: str
    tagline: str
    description: str
    instinct: str
    confidence: float = Field(..., ge=0.0, le=100.0)
    confidence_delta: float        # pts vs yesterday
    confidence_persistence: int    # days above 80%
    duration_days: int
    last_changed: datetime
    drivers: Dict[str, DriverScore]
    timestamp: datetime


class ExposureRow(BaseModel):
    asset_class: str
    signal: str
    direction: str   # "up2" | "up" | "--" | "dn" | "dn2"
    magnitude: str
    confidence: float


class ExposureMap(BaseModel):
    regime: RegimeCode
    subtitle: str
    rows: List[ExposureRow]
    timestamp: datetime


class Narrative(BaseModel):
    regime: RegimeCode
    subtitle: str
    text: str
    horizon: str       # "Short" | "Medium"
    risk_flag: str
    timestamp: datetime


class TransitionProb(BaseModel):
    regime: RegimeCode
    probability: float


class RegimeSummary(BaseModel):
    state: RegimeState
    exposure: ExposureMap
    narrative: Narrative
    transitions: List[TransitionProb]


class HistoryPoint(BaseModel):
    date: datetime
    regime: RegimeCode
    confidence: float
    growth_score: float
    inflation_score: float
    liquidity_score: float
    volatility_score: float


class SimulateRequest(BaseModel):
    growth: float = Field(..., ge=-2.0, le=2.0, description="Growth driver score")
    inflation: float = Field(..., ge=-2.0, le=2.0, description="Inflation driver score")
    liquidity: float = Field(..., ge=-2.0, le=2.0, description="Liquidity driver score")
    volatility: float = Field(..., ge=-2.0, le=2.0, description="Volatility driver score")
