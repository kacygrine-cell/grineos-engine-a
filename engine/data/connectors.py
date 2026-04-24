"""
Data Connector Stubs
====================
Plug-in interfaces for live market data sources.

Each connector returns normalised driver scores on [-2, +2].
Scores are z-normalised against a 2-year rolling window.

To activate a connector:
  1. Set the relevant env vars (see each class)
  2. Replace the stub body with real fetch logic
  3. Set USE_LIVE_DATA=true in environment

Available connectors:
  - FREDConnector     : St. Louis Fed FRED API (GDP, CPI, M2)
  - YahooConnector    : Yahoo Finance (VIX, equities, commodities)
  - FedConnector      : Federal Reserve H.15 (rates, spreads)
  - BloombergConnector: Bloomberg (stub only — requires licence)
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
from datetime import datetime


# ── Base interface ─────────────────────────────────────────────────────────────
class BaseConnector:
    """
    All connectors must implement fetch().
    Returns (current_scores, previous_scores) as raw score dicts.
    Keys: "growth", "inflation", "liquidity", "volatility"
    """
    NAME = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def fetch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        raise NotImplementedError


# ── FRED Connector ─────────────────────────────────────────────────────────────
class FREDConnector(BaseConnector):
    """
    St. Louis Federal Reserve FRED API.
    Env vars: FRED_API_KEY

    Series used:
      - Growth:    GDPC1 (real GDP), INDPRO (industrial production), ISM PMI
      - Inflation: CPIAUCSL, PPIACO, T5YIE (5yr inflation expectations)
      - Liquidity: M2SL (M2 money supply), DPCREDIT (discount window)
    """
    NAME = "FRED"

    def is_available(self) -> bool:
        return bool(os.getenv("FRED_API_KEY"))

    def fetch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        TODO: implement using fredapi or requests.

        Example:
            from fredapi import Fred
            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            gdp = fred.get_series("GDPC1")
            gdp_zscore = z_normalise(gdp, window=104)  # 2yr weekly
            ...
        """
        raise NotImplementedError("Set FRED_API_KEY and implement fetch()")


# ── Yahoo Finance Connector ────────────────────────────────────────────────────
class YahooConnector(BaseConnector):
    """
    Yahoo Finance via yfinance.
    No API key required.

    Tickers used:
      - Volatility: ^VIX (equity vol), ^MOVE (bond vol)
      - Growth proxy: SPY momentum, ^GSPC earnings revisions
      - Liquidity proxy: HYG/LQD spread, TLT
      - Commodities: GLD, USO
    """
    NAME = "Yahoo"

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa
            return True
        except ImportError:
            return False

    def fetch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        TODO: implement using yfinance.

        Example:
            import yfinance as yf
            vix = yf.download("^VIX", period="2y")["Close"]
            vix_zscore = z_normalise(vix)
            vol_score = -vix_zscore.iloc[-1]  # invert: high VIX = negative vol score
            ...
        """
        raise NotImplementedError("Install yfinance and implement fetch()")


# ── Federal Reserve H.15 Connector ────────────────────────────────────────────
class FedConnector(BaseConnector):
    """
    Federal Reserve H.15 Selected Interest Rates.
    No API key required (public data).

    Data used:
      - Liquidity: Fed Funds Rate vs 2yr Treasury (spread)
      - Growth: Yield curve (10yr - 2yr spread)
    URL: https://www.federalreserve.gov/releases/h15/
    """
    NAME = "Fed"

    def is_available(self) -> bool:
        return True  # Public endpoint, always available

    def fetch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        TODO: implement using requests + pandas.

        Example:
            import requests, pandas as pd
            url = "https://www.federalreserve.gov/releases/h15/H15.csv"
            df = pd.read_csv(url, skiprows=5)
            yield_curve = df["RIFLPBCIANM10Y_N.B"] - df["RIFLPBCIANM02Y_N.B"]
            lq_score = z_normalise(yield_curve).iloc[-1]
            ...
        """
        raise NotImplementedError("Implement Fed H.15 fetch()")


# ── Bloomberg Connector (stub) ─────────────────────────────────────────────────
class BloombergConnector(BaseConnector):
    """
    Bloomberg Terminal API (blpapi).
    Requires: Bloomberg licence + blpapi Python SDK.
    Env vars: BLOOMBERG_HOST, BLOOMBERG_PORT

    Fields used:
      - VIX Index (volatility)
      - MOVE Index (bond volatility)
      - US0003M Index (LIBOR/SOFR proxy)
      - SPX Index (equity momentum)
      - USGG10YR Index (yield curve)
      - HY OAS (credit spreads)
    """
    NAME = "Bloomberg"

    def is_available(self) -> bool:
        try:
            import blpapi  # noqa
            return bool(os.getenv("BLOOMBERG_HOST"))
        except ImportError:
            return False

    def fetch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        raise NotImplementedError("Bloomberg connector requires blpapi licence")


# ── Connector factory ──────────────────────────────────────────────────────────
def z_normalise(series, window: int = 504) -> "pd.Series":
    """
    Rolling z-score normalisation.
    window: number of observations (504 = ~2yr daily)
    """
    import pandas as pd
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0, 1)


def get_live_drivers() -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
    """
    Try each connector in priority order.
    Returns None if no live connector is available.
    """
    connectors = [BloombergConnector(), FREDConnector(), YahooConnector()]
    for connector in connectors:
        if connector.is_available():
            try:
                return connector.fetch()
            except NotImplementedError:
                continue
            except Exception as e:
                print(f"[Engine A] Connector {connector.NAME} failed: {e}")
                continue
    return None
