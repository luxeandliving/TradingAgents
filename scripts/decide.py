"""Callable entrypoint for a single propagate() decision — the bridge news-gap-ml's
live_listener.py subprocesses into once its OR-gate trigger fires (trading-workspace
issue #18). Runs the full multi-agent graph for one ticker/date and prints exactly
one line of JSON to stdout: the decision, nothing else. All progress/error output
goes to stderr so a subprocess caller can parse stdout unconditionally.

This makes a real, billed call to the configured LLM provider — it is not free and
is not fast (propagate() chains several sequential LLM calls through the analyst/
debate/risk graph). Do not call this in a tight loop.

Ticker must include the exchange suffix TradingAgents expects (see README.md,
e.g. "RELIANCE.NS" for NSE India, "WIPRO.NS", plain "NVDA" for US) — this script
does not guess or normalize tickers.

Usage:
    python scripts/decide.py --ticker WIPRO.NS --date 2026-07-15
    python scripts/decide.py --ticker NVDA --date 2026-07-15 --asset-type stock

Output (stdout, single line):
    {"ticker": "WIPRO.NS", "trade_date": "2026-07-15", "rating": "Buy",
     "holding_recommendation": "Square Off Intraday",
     "final_trade_decision": "...", "generated_at": "2026-07-15T09:03:11+00:00",
     "cost_usd": 0.0412, "token_usage": {"claude-sonnet-4-6": {"input_tokens": 8000, ...}}}

On failure: non-zero exit code, error detail on stderr, nothing on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from langchain_core.callbacks import UsageMetadataCallbackHandler

from tradingagents.agents.utils.rating import parse_holding_recommendation
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Indian tickers get much better news coverage from the indian_news vendor than
# the yfinance-only default (see default_config.py's data_vendors comment) —
# applied automatically so callers don't need to know this vendor detail.
_INDIAN_SUFFIXES = (".NS", ".BO")

# $ per 1M tokens (input, output) — trading-workspace issue #24. Keyed by the
# model_name langchain reports in AIMessage.response_metadata, which can carry
# a dated suffix (e.g. "claude-haiku-4-5-20251001") — _price_for() matches by
# longest-prefix so both bare and dated IDs resolve. Update when the .env
# TRADINGAGENTS_DEEP_THINK_LLM/QUICK_THINK_LLM models or their pricing change.
_PRICING_PER_MTOK = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
}


def _price_for(model_name: str) -> tuple:
    """Longest-prefix match against _PRICING_PER_MTOK, or None if unknown."""
    match = max(
        (key for key in _PRICING_PER_MTOK if model_name.startswith(key)),
        key=len, default=None,
    )
    return _PRICING_PER_MTOK[match] if match else None


def _compute_cost(usage_metadata: dict) -> tuple[float, list]:
    """Returns (total_usd, [model names with no pricing entry])."""
    total = 0.0
    unpriced = []
    for model_name, usage in usage_metadata.items():
        price = _price_for(model_name)
        if price is None:
            unpriced.append(model_name)
            continue
        price_in, price_out = price
        total += usage.get("input_tokens", 0) / 1_000_000 * price_in
        total += usage.get("output_tokens", 0) / 1_000_000 * price_out
    return round(total, 4), unpriced


def _build_config(ticker: str) -> dict:
    config = DEFAULT_CONFIG.copy()
    if ticker.upper().endswith(_INDIAN_SUFFIXES):
        config["data_vendors"] = {
            **config["data_vendors"],
            "news_data": "indian_news,yfinance",
        }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True, help='e.g. "WIPRO.NS", "RELIANCE.NS", "NVDA"')
    parser.add_argument("--date", required=True, dest="trade_date", help="YYYY-MM-DD")
    parser.add_argument("--asset-type", default="stock", choices=["stock", "crypto"])
    args = parser.parse_args()

    usage_handler = UsageMetadataCallbackHandler()
    try:
        config = _build_config(args.ticker)
        ta = TradingAgentsGraph(debug=False, config=config, callbacks=[usage_handler])
        final_state, rating = ta.propagate(args.ticker, args.trade_date, asset_type=args.asset_type)
        final_decision = final_state["final_trade_decision"]
        holding_recommendation = parse_holding_recommendation(final_decision)
    except Exception as exc:  # noqa: BLE001 — report cleanly on stderr, never on stdout
        print(f"decide.py failed for {args.ticker} on {args.trade_date}: {exc}", file=sys.stderr)
        return 1

    cost_usd, unpriced_models = _compute_cost(usage_handler.usage_metadata)
    if unpriced_models:
        print(f"decide.py: no pricing entry for model(s) {unpriced_models} — cost_usd is a partial total", file=sys.stderr)
    print(f"decide.py: ${cost_usd:.4f} for this call ({usage_handler.usage_metadata})", file=sys.stderr)

    result = {
        "ticker": args.ticker,
        "trade_date": args.trade_date,
        "asset_type": args.asset_type,
        "rating": rating,
        "holding_recommendation": holding_recommendation,
        "final_trade_decision": final_decision,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cost_usd": cost_usd,
        "token_usage": usage_handler.usage_metadata,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
