"""Tests for the deterministic scorer behind decision_mode="structured"
(trading-workspace TradingAgents#19). Pure functions, no LLM/API calls."""
import pytest

from tradingagents.agents.managers.decision_model import compute_rating, score_factors
from tradingagents.agents.schemas import FactorExtraction, PortfolioRating


def _factors(**overrides):
    base = {
        "dated_catalyst_present": True,
        "catalyst_hours_to_resolution": 4.0,
        "technical_direction": 0.0,
        "technical_confidence": 0.0,
        "sentiment_direction": 0.0,
        "sentiment_confidence": 0.0,
        "risk_flags": [],
    }
    base.update(overrides)
    return FactorExtraction(**base)


@pytest.mark.unit
class TestCatalystHardGate:
    def test_no_catalyst_forces_hold_regardless_of_strong_scores(self):
        factors = _factors(
            dated_catalyst_present=False,
            technical_direction=1.0, technical_confidence=1.0,
            sentiment_direction=1.0, sentiment_confidence=1.0,
        )
        rating, score, reason = compute_rating(factors)
        assert rating == PortfolioRating.HOLD
        assert score == 0.0
        assert "No dated catalyst" in reason

    def test_catalyst_too_far_out_forces_hold(self):
        factors = _factors(
            catalyst_hours_to_resolution=48.0,
            technical_direction=1.0, technical_confidence=1.0,
        )
        rating, score, reason = compute_rating(factors)
        assert rating == PortfolioRating.HOLD
        assert "48.0h" in reason

    def test_catalyst_within_window_does_not_force_hold(self):
        factors = _factors(
            catalyst_hours_to_resolution=23.9,
            technical_direction=1.0, technical_confidence=1.0,
        )
        rating, score, _ = compute_rating(factors)
        assert rating != PortfolioRating.HOLD or score != 0.0

    def test_catalyst_with_unknown_timing_does_not_force_hold(self):
        """catalyst_hours_to_resolution=None (timing unclear) should not be
        treated the same as "too far out" -- only an explicit large value gates."""
        factors = _factors(
            catalyst_hours_to_resolution=None,
            technical_direction=1.0, technical_confidence=1.0,
        )
        rating, score, _ = compute_rating(factors)
        assert rating == PortfolioRating.BUY


@pytest.mark.unit
class TestScoreThresholds:
    def test_strong_bullish_technical_and_sentiment_yields_buy(self):
        factors = _factors(
            technical_direction=0.9, technical_confidence=0.9,
            sentiment_direction=0.9, sentiment_confidence=0.9,
        )
        rating, score, _ = compute_rating(factors)
        assert rating == PortfolioRating.BUY
        assert score >= 0.5

    def test_strong_bearish_yields_sell(self):
        factors = _factors(
            technical_direction=-0.9, technical_confidence=0.9,
            sentiment_direction=-0.9, sentiment_confidence=0.9,
        )
        rating, score, _ = compute_rating(factors)
        assert rating == PortfolioRating.SELL

    def test_moderate_bullish_yields_overweight_not_buy(self):
        factors = _factors(technical_direction=0.5, technical_confidence=0.6)
        rating, score, _ = compute_rating(factors)
        assert score == pytest.approx(0.3)
        assert rating == PortfolioRating.OVERWEIGHT

    def test_low_confidence_single_source_is_suppressed_toward_hold(self):
        """A strong direction from a near-zero-confidence source must NOT
        reach a directional threshold -- confidence has to actually damp the
        score, not just weight it relative to a second (absent) source."""
        factors = _factors(technical_direction=0.9, technical_confidence=0.05)
        rating, score, _ = compute_rating(factors)
        assert score == pytest.approx(0.045)
        assert rating == PortfolioRating.HOLD

    def test_two_aligned_confident_sources_reinforce_and_clamp_at_cap(self):
        factors = _factors(
            technical_direction=0.8, technical_confidence=0.9,
            sentiment_direction=0.8, sentiment_confidence=0.9,
        )
        rating, score, _ = compute_rating(factors)
        assert score == 1.0  # 0.72 + 0.72 = 1.44, clamped
        assert rating == PortfolioRating.BUY

    def test_zero_confidence_on_both_sides_yields_hold(self):
        """No basis to lean either way -- zero confidence must not divide by
        zero or produce a spurious directional call."""
        factors = _factors(
            technical_direction=1.0, technical_confidence=0.0,
            sentiment_direction=-1.0, sentiment_confidence=0.0,
        )
        rating, score, _ = compute_rating(factors)
        assert rating == PortfolioRating.HOLD
        assert score == 0.0

    def test_conflicting_technical_and_sentiment_partially_cancel(self):
        """Equal-confidence opposite signals should net toward Hold, not
        double-count either side."""
        factors = _factors(
            technical_direction=1.0, technical_confidence=1.0,
            sentiment_direction=-1.0, sentiment_confidence=1.0,
        )
        rating, score, _ = compute_rating(factors)
        assert score == 0.0
        assert rating == PortfolioRating.HOLD


@pytest.mark.unit
class TestRiskFlagDamping:
    def test_risk_flags_pull_a_borderline_buy_toward_hold(self):
        factors_clean = _factors(technical_direction=0.55, technical_confidence=0.6)
        factors_flagged = _factors(
            technical_direction=0.55, technical_confidence=0.6,
            risk_flags=["earnings in window", "illiquid", "contested by analysts"],
        )
        _, score_clean, _ = compute_rating(factors_clean)
        _, score_flagged, _ = compute_rating(factors_flagged)
        assert abs(score_flagged) < abs(score_clean)

    def test_many_risk_flags_can_flip_overweight_to_hold(self):
        factors = _factors(
            technical_direction=0.25, technical_confidence=0.9,
            risk_flags=["a", "b", "c", "d", "e", "f", "g"],  # 7 * 0.15 = 1.05 damping -> 0
        )
        rating, score, _ = compute_rating(factors)
        assert score == 0.0
        assert rating == PortfolioRating.HOLD


@pytest.mark.unit
class TestScoreFactorsRendersResearchPlan:
    def test_hold_plan_has_no_position_language(self):
        plan_md = score_factors(_factors(dated_catalyst_present=False), company_name="NVDA")
        assert "**Recommendation**: Hold" in plan_md
        assert "stand aside" in plan_md.lower()

    def test_buy_plan_names_long_direction_and_company(self):
        factors = _factors(technical_direction=0.9, technical_confidence=0.9)
        plan_md = score_factors(factors, company_name="NVDA")
        assert "**Recommendation**: Buy" in plan_md
        assert "long" in plan_md.lower()
        assert "NVDA" in plan_md

    def test_sell_plan_names_short_direction(self):
        factors = _factors(technical_direction=-0.9, technical_confidence=0.9)
        plan_md = score_factors(factors, company_name="NVDA")
        assert "short" in plan_md.lower()
