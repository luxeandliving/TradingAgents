"""Deterministic scorer for the "structured" decision_mode (trading-workspace
TradingAgents#19).

Fixed, versioned, code-reviewed function -- NOT an LLM call. Takes the
FactorExtraction produced by factor_extractor.py and maps it to a
ResearchPlan the same way the Research Manager's debate synthesis would,
via an auditable formula instead of free-form argument. The LLM upstream can
adjust the *factors*; it never sees or can influence this function.

v0 is a simple weighted-average + fixed-threshold model, deliberately not
tuned yet -- there isn't enough decision_outcomes data to fit real weights
against (news-gap-ml's decision_outcomes table has 138 decisions / 72 scored
verdicts as of 2026-08-29). Treat every threshold below as a starting point
to be recalibrated once the retro-validation batch produces real outcomes,
not as a considered final design.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioRating, ResearchPlan, render_research_plan

# The whole strategy this framework targets is a close-to-open gap trade --
# every retro transcript in this workspace's validation history converges on
# "no dated catalyst inside the decision window -> Hold" regardless of how
# strong the technical/sentiment case looks otherwise. Encode that as a hard
# gate rather than hoping the weighted score naturally lands on Hold.
_MAX_CATALYST_HOURS = 24.0

# Rating thresholds on the combined, confidence-weighted directional score
# (range roughly -1..+1). Asymmetric bands are not implied by anything other
# than matching the enum's 5 tiers evenly -- recalibrate once real outcomes
# exist.
_BUY_THRESHOLD = 0.5
_OVERWEIGHT_THRESHOLD = 0.2
_UNDERWEIGHT_THRESHOLD = -0.2
_SELL_THRESHOLD = -0.5

# Each risk flag shaves this fraction off the combined score's magnitude
# (pulls toward Hold) rather than gating outright -- a single flag shouldn't
# override a strong, catalyst-backed case, but several should.
_RISK_FLAG_DAMPING_PER_FLAG = 0.15


def _combined_score(factors) -> float:
    """Confidence-weighted sum of the technical and sentiment directions,
    clamped to [-1, 1].

    Deliberately a weighted SUM, not an average: an average normalizes by
    weight_sum, which means a single near-zero-confidence source still fully
    determines the score at its raw direction value (e.g. direction=0.9,
    confidence=0.01 would score 0.9 under an average -- confidence would only
    ever matter for weighting *between* two sources, never for suppressing a
    single low-confidence one). A weighted sum fixes that: low confidence
    genuinely damps toward zero regardless of how many sources are present,
    while a single fully-confident source can still reach full conviction
    (direction=1, confidence=1 -> 1.0) and two aligned confident sources can
    reinforce past what either alone would reach (clamped at the cap)."""
    raw = (
        factors.technical_direction * factors.technical_confidence
        + factors.sentiment_direction * factors.sentiment_confidence
    )
    return max(-1.0, min(1.0, raw))


def _rating_for_score(score: float) -> PortfolioRating:
    if score >= _BUY_THRESHOLD:
        return PortfolioRating.BUY
    if score >= _OVERWEIGHT_THRESHOLD:
        return PortfolioRating.OVERWEIGHT
    if score <= _SELL_THRESHOLD:
        return PortfolioRating.SELL
    if score <= _UNDERWEIGHT_THRESHOLD:
        return PortfolioRating.UNDERWEIGHT
    return PortfolioRating.HOLD


def compute_rating(factors) -> tuple[PortfolioRating, float, str]:
    """Returns (rating, final_score, reason) -- the pure decision logic,
    separated from ResearchPlan rendering so it's directly unit-testable."""
    if not factors.dated_catalyst_present:
        return (
            PortfolioRating.HOLD, 0.0,
            "No dated catalyst identified within this decision's holding window -- "
            "hard gate to Hold regardless of technical/sentiment scores.",
        )
    if (
        factors.catalyst_hours_to_resolution is not None
        and factors.catalyst_hours_to_resolution > _MAX_CATALYST_HOURS
    ):
        return (
            PortfolioRating.HOLD, 0.0,
            f"Catalyst identified but resolves in {factors.catalyst_hours_to_resolution:.1f}h, "
            f"beyond the {_MAX_CATALYST_HOURS:.0f}h window this close-to-open strategy targets -- "
            "hard gate to Hold.",
        )

    score = _combined_score(factors)
    n_flags = len(factors.risk_flags)
    if n_flags:
        damping = max(0.0, 1.0 - _RISK_FLAG_DAMPING_PER_FLAG * n_flags)
        score *= damping

    rating = _rating_for_score(score)
    reason = (
        f"Combined confidence-weighted score {score:+.2f} "
        f"(technical {factors.technical_direction:+.2f}@{factors.technical_confidence:.2f} conf, "
        f"sentiment {factors.sentiment_direction:+.2f}@{factors.sentiment_confidence:.2f} conf"
        + (f", damped {n_flags} risk flag(s)" if n_flags else "")
        + f") -> {rating.value}."
    )
    return rating, score, reason


def score_factors(factors, company_name: str) -> str:
    """Maps FactorExtraction -> a rendered ResearchPlan string, the same
    shape state["investment_plan"] already holds for the debate_enabled=True
    path, so Trader/Portfolio Manager consume it unchanged."""
    rating, score, reason = compute_rating(factors)

    rationale_parts = [reason]
    if factors.dated_catalyst_present and factors.catalyst_hours_to_resolution is not None:
        rationale_parts.append(
            f"Catalyst expected to resolve in ~{factors.catalyst_hours_to_resolution:.1f}h."
        )
    if factors.risk_flags:
        rationale_parts.append(f"Risk flags: {', '.join(factors.risk_flags)}.")

    if rating == PortfolioRating.HOLD:
        strategic_actions = "No fresh position -- stand aside for this decision window."
    else:
        direction = "long" if score > 0 else "short"
        strategic_actions = (
            f"Open a {direction}-biased options position in {company_name} sized to the "
            f"{rating.value} conviction level (score {score:+.2f}); size down if any risk "
            f"flags are present. Standard close-to-open exit discipline applies."
        )

    plan = ResearchPlan(
        recommendation=rating,
        rationale=" ".join(rationale_parts),
        strategic_actions=strategic_actions,
    )
    return render_research_plan(plan)
