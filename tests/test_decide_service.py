"""Tests for the TradingAgents HTTP decision service (hermes#213) --
scripts/service.py, and decide.py's run_decision() extraction that both the
CLI and the service share.

Context: news-gap-ml's containerized cron can no longer subprocess into this
venv directly (container filesystem isolation broke the sibling-directory
assumption -- every triggered decision silently failed from the 2026-07-15/16
container cutover onward). This service is the fix: news-gap-ml reaches
TradingAgents over a loopback HTTP call instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


def _load_module(name: str, filename: str):
    """scripts/ isn't a package (no __init__.py) -- load a script module
    directly by path, same pattern test_external_signal_context.py already
    uses for decide.py."""
    path = Path(__file__).parent.parent / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def service():
    return _load_module("tradingagents_service_under_test", "service.py")


@pytest.fixture
def client(service):
    return TestClient(service.app)


@pytest.mark.unit
class TestRunDecisionExtraction:
    """run_decision() must behave identically to decide.py's old inline
    main() body -- this is a pure extract-function refactor."""

    def test_run_decision_returns_expected_shape(self):
        decide = _load_module("tradingagents_decide_under_test", "decide.py")
        fake_final_state = {"final_trade_decision": "FINAL TRANSACTION PROPOSAL: **BUY**"}
        with patch.object(decide, "TradingAgentsGraph") as MockGraph:
            MockGraph.return_value.propagate.return_value = (fake_final_state, "Buy")
            result = decide.run_decision("WIPRO.NS", "2026-07-16")

        assert result["ticker"] == "WIPRO.NS"
        assert result["trade_date"] == "2026-07-16"
        assert result["asset_type"] == "stock"
        assert result["rating"] == "Buy"
        assert result["final_trade_decision"] == fake_final_state["final_trade_decision"]
        assert "generated_at" in result
        assert "cost_usd" in result
        assert "token_usage" in result

    def test_run_decision_raises_on_failure(self):
        """CLI catches this and exits 1; the service catches it and returns 500."""
        decide = _load_module("tradingagents_decide_under_test", "decide.py")
        with patch.object(decide, "TradingAgentsGraph") as MockGraph:
            MockGraph.return_value.propagate.side_effect = RuntimeError("LLM provider timeout")
            with pytest.raises(RuntimeError, match="LLM provider timeout"):
                decide.run_decision("WIPRO.NS", "2026-07-16")


@pytest.mark.unit
class TestServiceAuth:
    def test_health_requires_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_decide_rejects_missing_auth_header(self, client, service):
        service._SERVICE_SECRET = "real-secret"
        resp = client.post("/decide", json={"ticker": "WIPRO.NS", "trade_date": "2026-07-16"})
        assert resp.status_code == 403

    def test_decide_rejects_wrong_secret(self, client, service):
        service._SERVICE_SECRET = "real-secret"
        resp = client.post(
            "/decide", json={"ticker": "WIPRO.NS", "trade_date": "2026-07-16"},
            headers={"Authorization": "Bearer wrong-secret"},
        )
        assert resp.status_code == 403

    def test_decide_fails_closed_when_secret_unconfigured(self, client, service):
        """An unset TRADINGAGENTS_SERVICE_SECRET must reject every request --
        never silently skip auth (same posture as hermes' _validate_webhook,
        issue #144)."""
        service._SERVICE_SECRET = ""
        resp = client.post(
            "/decide", json={"ticker": "WIPRO.NS", "trade_date": "2026-07-16"},
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 403

    def test_decide_accepts_correct_secret(self, client, service):
        service._SERVICE_SECRET = "real-secret"
        with patch.object(service, "run_decision", return_value={"rating": "Hold"}):
            resp = client.post(
                "/decide", json={"ticker": "WIPRO.NS", "trade_date": "2026-07-16"},
                headers={"Authorization": "Bearer real-secret"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"rating": "Hold"}


@pytest.mark.unit
class TestServiceDecideEndpoint:
    def test_decide_passes_request_fields_through(self, client, service):
        service._SERVICE_SECRET = "s"
        with patch.object(service, "run_decision", return_value={"rating": "Sell"}) as mock_run:
            client.post(
                "/decide",
                json={
                    "ticker": "WIPRO.NS", "trade_date": "2026-07-16",
                    "asset_type": "stock", "context": '{"side": "short"}',
                },
                headers={"Authorization": "Bearer s"},
            )
        mock_run.assert_called_once_with("WIPRO.NS", "2026-07-16", "stock", '{"side": "short"}')

    def test_decide_defaults_asset_type_and_context(self, client, service):
        service._SERVICE_SECRET = "s"
        with patch.object(service, "run_decision", return_value={"rating": "Hold"}) as mock_run:
            client.post(
                "/decide", json={"ticker": "NVDA", "trade_date": "2026-07-16"},
                headers={"Authorization": "Bearer s"},
            )
        mock_run.assert_called_once_with("NVDA", "2026-07-16", "stock", None)

    def test_decide_surfaces_failure_as_500(self, client, service):
        service._SERVICE_SECRET = "s"
        with patch.object(service, "run_decision", side_effect=RuntimeError("LLM provider timeout")):
            resp = client.post(
                "/decide", json={"ticker": "WIPRO.NS", "trade_date": "2026-07-16"},
                headers={"Authorization": "Bearer s"},
            )
        assert resp.status_code == 500
        assert "LLM provider timeout" in resp.json()["detail"]

    def test_decide_missing_required_field_is_422(self, client, service):
        service._SERVICE_SECRET = "s"
        resp = client.post(
            "/decide", json={"ticker": "WIPRO.NS"},  # missing trade_date
            headers={"Authorization": "Bearer s"},
        )
        assert resp.status_code == 422
