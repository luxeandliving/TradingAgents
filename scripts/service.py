"""HTTP service wrapper around decide.py's run_decision() (hermes#213).

news-gap-ml's containerized cron can no longer subprocess into this venv
directly -- container filesystem isolation breaks the sibling-directory
assumption call_tradingagents_decision() relied on (TRADINGAGENTS_DIR
resolves to a path that doesn't exist inside the container). This exposes
the same decision-making contract over a local HTTP endpoint instead: both
processes run on the same droplet, so this is a loopback call, not a real
network boundary crossing.

Auth: a single shared secret (TRADINGAGENTS_SERVICE_SECRET), same
fail-closed / constant-time-compare posture as MOTILAL_WEBHOOK_SECRET
elsewhere in this workspace -- internal-only, both sides on the same box.
Sent as a Bearer token rather than embedded in the JSON body, since
(unlike the TradingView-originated broker webhooks that pattern comes
from) this isn't a third-party source whose request shape we don't
control.

Run with: uvicorn scripts.service:app --host 127.0.0.1 --port 8100
(loopback-only -- this must never be exposed off the box, the shared
secret is not a substitute for network isolation).
"""
from __future__ import annotations

import hmac
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# scripts/ has no __init__.py (it's a collection of standalone entrypoints,
# not a package) -- add it to sys.path so `from decide import run_decision`
# resolves the same way running decide.py directly would.
sys.path.insert(0, str(Path(__file__).parent))
from decide import run_decision  # noqa: E402

_SERVICE_SECRET = os.getenv("TRADINGAGENTS_SERVICE_SECRET", "")

app = FastAPI(title="TradingAgents decision service")


class DecisionRequest(BaseModel):
    ticker: str
    trade_date: str
    asset_type: str = "stock"
    context: str | None = None


def check_auth(authorization: str | None) -> None:
    """Fail closed: an unconfigured secret rejects every request rather
    than skipping validation entirely, same posture as hermes'
    _validate_webhook() (#144) -- an unset secret must never mean
    "unauthenticated," it must mean "nothing gets in."
    """
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not _SERVICE_SECRET or not hmac.compare_digest(token, _SERVICE_SECRET):
        raise HTTPException(status_code=403, detail="invalid or missing service secret")


@app.get("/health")
def health() -> dict:
    """No auth -- used for the deploy smoke check and hermes#218's cross-service
    health page, carries no sensitive data."""
    return {"status": "ok", "version": os.getenv("APP_VERSION", "dev")}


@app.post("/decide")
def decide(req: DecisionRequest, authorization: str | None = Header(default=None)) -> dict:
    check_auth(authorization)
    try:
        return run_decision(req.ticker, req.trade_date, req.asset_type, req.context)
    except Exception as exc:  # noqa: BLE001 -- surface as a clean 500, same info decide.py puts on stderr
        raise HTTPException(
            status_code=500,
            detail=f"decide.py failed for {req.ticker} on {req.trade_date}: {exc}",
        ) from exc
