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
     "final_trade_decision": "...", "generated_at": "2026-07-15T09:03:11+00:00"}

On failure: non-zero exit code, error detail on stderr, nothing on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Indian tickers get much better news coverage from the indian_news vendor than
# the yfinance-only default (see default_config.py's data_vendors comment) —
# applied automatically so callers don't need to know this vendor detail.
_INDIAN_SUFFIXES = (".NS", ".BO")


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

    try:
        config = _build_config(args.ticker)
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, rating = ta.propagate(args.ticker, args.trade_date, asset_type=args.asset_type)
        final_decision = final_state["final_trade_decision"]
    except Exception as exc:  # noqa: BLE001 — report cleanly on stderr, never on stdout
        print(f"decide.py failed for {args.ticker} on {args.trade_date}: {exc}", file=sys.stderr)
        return 1

    result = {
        "ticker": args.ticker,
        "trade_date": args.trade_date,
        "asset_type": args.asset_type,
        "rating": rating,
        "final_trade_decision": final_decision,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
