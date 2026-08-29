"""Catalyst-date cohort test for decision_mode="structured" (TradingAgents#19).

The 10-symbol post-sync smoke cohort (retro_smoke_postsync.py) produced 10/10
Hold under decision_mode="structured", because its hard gate forces Hold
whenever no dated catalyst resolves within 24h -- and none of those 10
(ticker, date) pairs happened to sit next to a real catalyst. This script
picks trade_date = the trading day immediately before each ticker's real,
already-reported Q1 FY27 earnings date (per yfinance get_earnings_dates,
2026-08-29), so the catalyst hard gate has an actual near-term event to
find. Tests whether the "structured" directional branch can fire at all,
not just its Hold fallback.

Real, billed LLM calls via decide.py's run_decision() -- same production
entrypoint as the post-sync smoke test. ~$0.35-0.42/decision observed on
the prior structured-mode batch, so this ~9-call batch is expected to cost
roughly $3-4 and take ~55-65 min sequentially.
"""
import json
import os
import sys
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ["TRADINGAGENTS_DECISION_MODE"] = "structured"

# Norton SSL inspection breaks curl_cffi (yfinance's TLS stack) in this local
# environment -- same workaround as run_scanner_retro.py / retro_smoke_postsync.py.
try:
    import curl_cffi.requests as _cffi_req
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_session_init = _cffi_req.Session.__init__
    def _patched_session_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_session_init(self, *args, **kwargs)
    _cffi_req.Session.__init__ = _patched_session_init
    print("yfinance: curl_cffi Session patched with verify=False (Norton SSL workaround)")
except Exception as e:
    print(f"yfinance SSL patch failed: {e} -- downloads may fail")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.decide import run_decision  # noqa: E402

# trade_date = trading day before the real reported earnings date (yfinance
# get_earnings_dates, checked 2026-08-29). Earnings date shown in the comment
# for traceability.
ENTRIES = [
    ("TCS.NS",         "2026-07-07"),  # earnings 2026-07-08
    ("WIPRO.NS",       "2026-07-15"),  # earnings 2026-07-16
    ("RELIANCE.NS",    "2026-07-16"),  # earnings 2026-07-17
    ("HDFCBANK.NS",    "2026-07-17"),  # earnings 2026-07-18
    ("ICICIBANK.NS",   "2026-07-17"),  # earnings 2026-07-18
    ("AXISBANK.NS",    "2026-07-17"),  # earnings 2026-07-18
    ("INFY.NS",        "2026-07-22"),  # earnings 2026-07-23
    ("MARUTI.NS",      "2026-07-30"),  # earnings 2026-07-31
    ("SBIN.NS",        "2026-08-06"),  # earnings 2026-08-07
]

JSONL_FILE = "retro_catalyst_cohort_structured_results.jsonl"
MD_FILE = "retro_catalyst_cohort_structured_results.md"

md_lines = [
    "# Catalyst-Date Cohort (decision_mode=structured)",
    f"\n**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "\n**Purpose:** test whether the structured decision mode's directional "
    "branch (vs. its no-catalyst Hold hard-gate) can fire at all, using "
    "trade_dates chosen to sit immediately before each ticker's real, "
    "already-reported Q1 FY27 earnings date.",
    "\n---\n",
]

total_cost = 0.0
with open(JSONL_FILE, "w", encoding="utf-8") as jf:
    for ticker, date in ENTRIES:
        print(f"\n{'='*60}\nRunning: {ticker} @ {date}\n{'='*60}")
        sys.stdout.flush()
        try:
            result = run_decision(ticker, date)
            total_cost += result.get("cost_usd", 0.0)
            jf.write(json.dumps(result) + "\n")
            jf.flush()
            print(f"DECISION: {result['rating']} / {result['holding_recommendation']} "
                  f"(${result['cost_usd']:.4f})")
            md_lines.append(
                f"## {ticker} @ {date}\n"
                f"**Rating:** {result['rating']}  \n"
                f"**Holding:** {result['holding_recommendation']}  \n"
                f"**Cost:** ${result['cost_usd']:.4f}\n\n"
                f"{result['final_trade_decision']}\n"
            )
        except Exception as e:
            err = f"ERROR: {e}"
            print(err)
            jf.write(json.dumps({"ticker": ticker, "trade_date": date, "error": str(e)}) + "\n")
            jf.flush()
            md_lines.append(f"## {ticker} @ {date}\n**Error:** {e}\n")

        with open(MD_FILE, "w", encoding="utf-8") as mf:
            mf.write("\n".join(md_lines) + f"\n\n---\n**Running total cost:** ${total_cost:.4f}\n")
        print(f"Saved intermediate results to {MD_FILE} (running cost: ${total_cost:.4f})")

print(f"\nAll done. Total cost: ${total_cost:.4f}")
