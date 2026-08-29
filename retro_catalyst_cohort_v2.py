"""Second catalyst-date cohort test for decision_mode="structured" (TradingAgents#19/#21).

The first catalyst cohort (retro_catalyst_cohort.py: TCS/WIPRO/RELIANCE/
HDFCBANK/ICICIBANK/AXISBANK/INFY/MARUTI/SBIN) surfaced a real look-ahead-bias
bug in fundamentals data (fixed in #21/PR #22) and, once fixed, went 0-for-9
on ever producing a directional call -- partly legitimate (most of those
trade_dates genuinely had no resolvable catalyst) and partly confounded by
the pipeline's own cross-run memory log recalling the first (leaked) run's
Overweight calls on the second re-run.

This is a fully fresh set of 9 tickers -- none overlap with any prior batch
in this project -- so the memory log has nothing to recall, giving a clean
read on whether "structured" mode's directional branch can fire on a real,
non-leaked, dated catalyst now that #21 is fixed.

Real, billed LLM calls via decide.py's run_decision() -- same production
entrypoint as prior batches. ~$0.35-0.42/decision observed previously, so
this ~9-call batch is expected to cost roughly $3-4 and take ~55-65 min
sequentially.
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

# trade_date = last real trading day before each ticker's real reported
# earnings date (yfinance get_earnings_dates, checked 2026-08-29). Earnings
# date + weekday shown in the comment for traceability -- a couple report on
# a Saturday (market closed), so trade_date is the preceding Friday, not the
# calendar day before.
ENTRIES = [
    ("HCLTECH.NS",     "2026-07-10"),  # earnings 2026-07-13 (Mon)
    ("KOTAKBANK.NS",   "2026-07-17"),  # earnings 2026-07-18 (Sat)
    ("ULTRACEMCO.NS",  "2026-07-17"),  # earnings 2026-07-20 (Mon)
    ("LT.NS",          "2026-07-27"),  # earnings 2026-07-28
    ("ASIANPAINT.NS",  "2026-07-28"),  # earnings 2026-07-29
    ("BAJFINANCE.NS",  "2026-07-29"),  # earnings 2026-07-30
    ("ITC.NS",         "2026-07-30"),  # earnings 2026-07-31
    ("BHARTIARTL.NS",  "2026-08-03"),  # earnings 2026-08-04
    ("TITAN.NS",       "2026-08-06"),  # earnings 2026-08-07
]

JSONL_FILE = "retro_catalyst_cohort_v2_structured_results.jsonl"
MD_FILE = "retro_catalyst_cohort_v2_structured_results.md"

md_lines = [
    "# Catalyst-Date Cohort v2 -- fresh tickers, post-#21-fix (decision_mode=structured)",
    f"\n**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "\n**Purpose:** re-test whether the structured decision mode's directional "
    "branch can fire on a real, non-leaked, dated catalyst using a fully "
    "fresh set of tickers (no overlap with any prior batch in this project, "
    "avoiding the cross-run memory-log confound seen re-running the first "
    "catalyst cohort). Trade_dates sit immediately before each ticker's "
    "real, already-reported Q1 FY27 earnings date.",
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
