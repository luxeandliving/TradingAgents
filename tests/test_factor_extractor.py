"""Tests for the Factor Extractor node (decision_mode="structured",
trading-workspace TradingAgents#19). No real LLM calls -- mocked structured
output, same pattern as test_global_shock.py."""
from unittest.mock import MagicMock

import pytest

from tradingagents.agents.managers.factor_extractor import create_factor_extractor
from tradingagents.agents.schemas import FactorExtraction

_STATE = {
    "company_of_interest": "NVDA",
    "market_report": "RSI 70, overbought.",
    "sentiment_report": "Bullish social chatter.",
    "news_report": "Guidance raised, resolves in 2h.",
    "fundamentals_report": "P/E 45x.",
}


def _llm_returning(factors: FactorExtraction):
    structured = MagicMock()
    structured.invoke.return_value = factors
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestFactorExtractorNode:
    def test_extracted_factors_and_investment_plan_are_populated(self):
        factors = FactorExtraction(
            dated_catalyst_present=True, catalyst_hours_to_resolution=2.0,
            technical_direction=0.8, technical_confidence=0.8,
            sentiment_direction=0.7, sentiment_confidence=0.7,
        )
        node = create_factor_extractor(_llm_returning(factors))
        result = node(_STATE)
        assert result["extracted_factors"] is factors
        assert "**Recommendation**: Buy" in result["investment_plan"]

    def test_structured_output_failure_falls_back_to_neutral_hold(self):
        """Must NOT fall back to free-text -- decision_model.py needs an
        actual FactorExtraction object, not a rendered string."""
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("provider error")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        node = create_factor_extractor(llm)

        result = node(_STATE)

        assert isinstance(result["extracted_factors"], FactorExtraction)
        assert result["extracted_factors"].dated_catalyst_present is False
        assert "**Recommendation**: Hold" in result["investment_plan"]

    def test_provider_without_structured_output_support_falls_back_to_neutral(self):
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError
        node = create_factor_extractor(llm)

        result = node(_STATE)

        assert isinstance(result["extracted_factors"], FactorExtraction)
        assert "**Recommendation**: Hold" in result["investment_plan"]
