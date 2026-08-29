"""Look-ahead-bias fix for retro/backtest fundamentals data (TradingAgents#19
structured-mode catalyst cohort, 2026-08-29; follow-up fix 2026-08-30).

yfinance financial statements are keyed by fiscal PERIOD-END date, not the
date results were actually announced. Indian companies typically report
2-4 weeks after quarter-end, so a naive "column date <= curr_date" filter
lets an unannounced quarter leak through as if it were already public --
confirmed live: ICICIBANK.NS's retro fundamentals report for 2026-07-17
(one day before its real 2026-07-18 Q1 FY27 announcement) already showed
that quarter's actual revenue/NIM figures. ``get_fundamentals`` was worse:
its ``curr_date`` param was documented as "not used for yfinance" and always
reflected today's live snapshot regardless of the requested retro date.

A first fix attempt (2026-08-29, PR #22) capped financial-statement columns
by a *count* of historically-announced quarters (``keep_at_most``). It
passed its own tests but still leaked live on ASIANPAINT.NS (2026-08-30):
the real count of a mature company's lifetime announced quarters (dozens)
vastly exceeds the handful of columns yfinance actually returns for
`quarterly_income_stmt` etc (~5), so the cap was silently a no-op in
practice. Corrected design: since quarterly cadence (~90 days) is always
much longer than reporting lag (2-4 weeks), at most ONE quarter can ever be
stuck in "ended but not yet announced" at a time, and it's necessarily the
newest surviving column -- so the fix is a boolean ``drop_newest`` driven by
``_unannounced_quarter_leaked()`` (a real announcement-date check), not a count.

These tests cover: `filter_financials_by_date`'s `drop_newest` param,
`get_fundamentals`'s embargo of quarter-dependent fields, and the specific
"many known quarters but few data columns" shape that broke the first
attempt.
"""
import unittest
from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.y_finance as y_finance
from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


def _earnings_dates_df(rows):
    """rows: list of (iso_date_str_with_tz_offset, reported_eps_or_none)."""
    index = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({"Reported EPS": [r[1] for r in rows]}, index=index)


# Mirrors a real get_earnings_dates() shape: newest first, future rows have
# NaN Reported EPS, past rows have a real value.
_ICICIBANK_LIKE_EARNINGS = _earnings_dates_df([
    ("2026-10-17 06:00:00-04:00", None),
    ("2026-07-18 04:00:00-04:00", 20.42),   # announced AFTER curr_date below
    ("2026-04-18 05:00:00-04:00", 18.90),
    ("2026-01-17 04:00:00-05:00", 15.62),
    ("2025-10-18 05:00:00-04:00", 17.06),
])

# The shape that broke the first fix attempt: a mature company with many
# years of real announced quarters (Reported EPS present), far more than
# yfinance's quarterly_income_stmt ever actually returns as columns (~5).
# All "reported" rows are strictly in the past relative to the curr_dates
# these tests use (2026-07-28/29) -- a future date can't already have a
# real Reported EPS, so none of the past-quarter rows may land after it.
_LONG_HISTORY_EARNINGS = _earnings_dates_df(
    [("2026-10-17 06:00:00-04:00", None), ("2026-07-29 04:00:00-04:00", 16.05)]
    + [(f"{2025 - (i // 4)}-{['01', '04', '07', '10'][i % 4]}-15 04:00:00-04:00", float(i))
       for i in range(22)]
)


@pytest.mark.unit
class FilterFinancialsByDateDropNewestTests(unittest.TestCase):
    def test_drop_newest_removes_only_the_single_newest_surviving_column(self):
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-06-30", "2026-03-31", "2025-12-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-28", drop_newest=True)
        self.assertEqual(list(out.columns), ["2026-03-31", "2025-12-31"])

    def test_drop_newest_false_preserves_prior_behavior(self):
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-06-30", "2026-03-31", "2025-12-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-28", drop_newest=False)
        self.assertEqual(list(out.columns), ["2026-06-30", "2026-03-31", "2025-12-31"])

    def test_drop_newest_on_empty_survivors_is_a_noop(self):
        data = pd.DataFrame(columns=["2026-06-30"])
        out = filter_financials_by_date(data.iloc[:, 0:0], "2026-07-28", drop_newest=True)
        self.assertEqual(list(out.columns), [])

    def test_date_mask_still_applies_before_the_drop(self):
        # A column strictly after curr_date is dropped by the existing date
        # mask regardless of drop_newest.
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-09-30", "2026-06-30", "2026-03-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-28", drop_newest=True)
        self.assertEqual(list(out.columns), ["2026-03-31"])

    def test_empty_data_or_no_curr_date_unaffected(self):
        data = pd.DataFrame({"2026-06-30": [1.0]})
        self.assertTrue(filter_financials_by_date(data, None, drop_newest=True).equals(data))
        empty = pd.DataFrame()
        self.assertTrue(filter_financials_by_date(empty, "2026-07-28", drop_newest=True).empty)


@pytest.mark.unit
class UnannouncedQuarterLeakTests(unittest.TestCase):
    def _mock_ticker(self, earnings_df):
        ticker_obj = mock.Mock()
        ticker_obj.get_earnings_dates.return_value = earnings_df
        return ticker_obj

    def test_true_when_real_announcement_is_after_curr_date(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        self.assertTrue(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-17"))

    def test_false_once_curr_date_is_on_or_after_it(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        self.assertFalse(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-18"))

    def test_false_on_fetch_failure(self):
        ticker_obj = mock.Mock()
        ticker_obj.get_earnings_dates.side_effect = RuntimeError("network down")
        self.assertFalse(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-17"))

    def test_true_for_a_mature_company_with_decades_of_reported_quarters(self):
        # Regression guard for the count-based approach's actual failure
        # mode: many real announced quarters in history is irrelevant to
        # whether THIS specific upcoming one has leaked.
        ticker_obj = self._mock_ticker(_LONG_HISTORY_EARNINGS)
        self.assertTrue(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-28"))
        self.assertFalse(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-29"))


@pytest.mark.unit
class GetFundamentalsEmbargoTests(unittest.TestCase):
    _INFO = {
        "longName": "ICICI Bank Limited",
        "sector": "Financial Services",
        "industry": "Banks",
        "marketCap": 10_210_000_000_000,
        "trailingPE": 18.2,
        "trailingEps": 78.5,
        "beta": 0.95,
        "fiftyTwoWeekHigh": 1500.0,
    }

    def _patch_ticker(self, earnings_df):
        ticker_obj = mock.Mock()
        ticker_obj.info = dict(self._INFO)
        ticker_obj.get_earnings_dates.return_value = earnings_df
        return mock.patch.object(y_finance.yf, "Ticker", return_value=ticker_obj)

    def test_quarter_sensitive_fields_omitted_when_announcement_lands_after_curr_date(self):
        with self._patch_ticker(_ICICIBANK_LIKE_EARNINGS):
            report = y_finance.get_fundamentals("ICICIBANK.NS", "2026-07-17")
        self.assertNotIn("PE Ratio (TTM)", report)
        self.assertNotIn("EPS (TTM)", report)
        self.assertIn("NOTE:", report)
        self.assertIn("2026-07-17", report)
        # Slow-moving profile fields survive the embargo.
        self.assertIn("Sector: Financial Services", report)
        self.assertIn("Beta: 0.95", report)

    def test_full_report_once_curr_date_is_on_or_after_the_real_announcement(self):
        with self._patch_ticker(_ICICIBANK_LIKE_EARNINGS):
            report = y_finance.get_fundamentals("ICICIBANK.NS", "2026-07-18")
        self.assertIn("PE Ratio (TTM)", report)
        self.assertIn("EPS (TTM)", report)
        self.assertNotIn("NOTE:", report)

    def test_no_curr_date_never_embargoes(self):
        with self._patch_ticker(_ICICIBANK_LIKE_EARNINGS):
            report = y_finance.get_fundamentals("ICICIBANK.NS", None)
        self.assertIn("PE Ratio (TTM)", report)
        self.assertNotIn("NOTE:", report)


@pytest.mark.unit
class GetIncomeStatementQuarterCapTests(unittest.TestCase):
    def test_unannounced_quarter_column_dropped_even_though_period_end_predates_curr_date(self):
        cols = [
            pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31"),
            pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30"),
        ]
        data = pd.DataFrame({c: [100.0] for c in cols})
        ticker_obj = mock.Mock()
        ticker_obj.quarterly_income_stmt = data
        ticker_obj.get_earnings_dates.return_value = _ICICIBANK_LIKE_EARNINGS

        with mock.patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            report = y_finance.get_income_statement("ICICIBANK.NS", "quarterly", "2026-07-17")

        self.assertNotIn("2026-06-30", report)
        self.assertIn("2026-03-31", report)
        self.assertIn("2025-12-31", report)
        self.assertIn("2025-09-30", report)

    def test_column_included_once_curr_date_reaches_the_real_announcement(self):
        cols = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31")]
        data = pd.DataFrame({c: [100.0] for c in cols})
        ticker_obj = mock.Mock()
        ticker_obj.quarterly_income_stmt = data
        ticker_obj.get_earnings_dates.return_value = _ICICIBANK_LIKE_EARNINGS

        with mock.patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            report = y_finance.get_income_statement("ICICIBANK.NS", "quarterly", "2026-07-18")

        self.assertIn("2026-06-30", report)

    def test_annual_frequency_never_checks_earnings_dates(self):
        # drop_newest is quarterly-earnings-specific; annual statements keep
        # the pre-existing date-mask-only behavior.
        cols = [pd.Timestamp("2026-03-31"), pd.Timestamp("2025-03-31")]
        data = pd.DataFrame({c: [100.0] for c in cols})
        ticker_obj = mock.Mock()
        ticker_obj.income_stmt = data
        ticker_obj.get_earnings_dates.return_value = _ICICIBANK_LIKE_EARNINGS

        with mock.patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            report = y_finance.get_income_statement("ICICIBANK.NS", "annual", "2026-07-17")

        self.assertIn("2026-03-31", report)
        self.assertIn("2025-03-31", report)
        ticker_obj.get_earnings_dates.assert_not_called()

    def test_mature_company_only_drops_the_single_leaked_column_not_all(self):
        # The exact shape that broke the first fix attempt (PR #22): a real
        # DataFrame with only 5 quarterly columns, but a much longer real
        # earnings-announcement history (22+ reported quarters) behind it.
        # A count-based cap (23 announced >= 5 columns) is a no-op here;
        # drop_newest correctly removes just the one leaked column.
        cols = [
            pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31"),
            pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30"),
            pd.Timestamp("2025-06-30"),
        ]
        data = pd.DataFrame({c: [100.0] for c in cols})
        ticker_obj = mock.Mock()
        ticker_obj.quarterly_income_stmt = data
        ticker_obj.get_earnings_dates.return_value = _LONG_HISTORY_EARNINGS

        with mock.patch.object(y_finance.yf, "Ticker", return_value=ticker_obj):
            report = y_finance.get_income_statement("ASIANPAINT.NS", "quarterly", "2026-07-28")

        self.assertNotIn("2026-06-30", report)
        self.assertIn("2026-03-31", report)
        self.assertIn("2025-06-30", report)


if __name__ == "__main__":
    unittest.main()
