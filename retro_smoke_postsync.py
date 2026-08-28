"""Post-sync smoke test (trading-workspace conversation 2026-08-28).

Re-runs the exact 10 (ticker, date) pairs used in retro_batch_10_v2 / _debate_off
(debate ON, default config) through the *current* main branch — which has since
pulled in "4 of 6 upstream fixes" (commit 8a6c258) — to confirm the production
decide.py entrypoint still behaves before committing to a larger retro-validation
batch. Uses scripts/decide.py's run_decision() directly (the same code path
news-gap-ml's live_listener.py calls), not a reimplementation.

Real, billed LLM calls -- ~$0.50/decision average per decide.py's docstring,
so this batch (10 calls) is expected to cost roughly $3-6 and take ~50-80 min
sequentially.
"""
import json
import os
import sys
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Norton SSL inspection breaks curl_cffi (yfinance's TLS stack) in this local
# environment -- same workaround as run_scanner_retro.py. Not needed on the
# droplet decide.py normally runs on, which is why decide.py itself lacks it.
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

# Identical inputs to retro_batch_10_v2_results.md / retro_batch_10_debate_off_results.md
ENTRIES = [
    ("ADANIPOWER.NS",  "2026-08-07"),
    ("PFC.NS",         "2026-08-07"),
    ("INFY.NS",        "2026-08-05"),
    ("PNB.NS",         "2026-07-28"),
    ("NMDC.NS",        "2026-07-28"),
    ("NBCC.NS",        "2026-07-28"),
    ("TMPV.NS",        "2026-07-28"),
    ("BDL.NS",         "2026-07-28"),
    ("PGEL.NS",        "2026-07-28"),
    ("FEDERALBNK.NS",  "2026-07-28"),
]

JSONL_FILE = "retro_smoke_postsync_results.jsonl"
MD_FILE = "retro_smoke_postsync_results.md"

md_lines = [
    "# Post-Sync Smoke Test (10 symbols, debate ON, default config)",
    f"\n**Run date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "\n**Purpose:** confirm current main (post commit 8a6c258 upstream sync) still "
    "produces sane decisions via the real production entrypoint (scripts/decide.py) "
    "before committing to a larger retro-validation batch. Identical (ticker, date) "
    "inputs to retro_batch_10_v2/_debate_off for direct comparison.",
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
