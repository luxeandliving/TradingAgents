"""Look-ahead-bias fix for retro/backtest fundamentals data (TradingAgents#19
structured-mode catalyst cohort, 2026-08-29).

yfinance financial statements are keyed by fiscal PERIOD-END date, not the
date results were actually announced. Indian companies typically report
2-4 weeks after quarter-end, so a naive "column date <= curr_date" filter
lets an unannounced quarter leak through as if it were already public --
confirmed live: ICICIBANK.NS's retro fundamentals report for 2026-07-17
(one day before its real 2026-07-18 Q1 FY27 announcement) already showed
that quarter's actual revenue/NIM figures. ``get_fundamentals`` was worse:
its ``curr_date`` param was documented as "not used for yfinance" and always
reflected today's live snapshot regardless of the requested retro date.

These tests cover the fix: `filter_financials_by_date`'s new `keep_at_most`
cap (driven by a real count of announced quarters, not period-end dates
alone), and `get_fundamentals`'s new embargo of quarter-dependent fields
when a real announcement landed after curr_date.
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


@pytest.mark.unit
class FilterFinancialsByDateKeepAtMostTests(unittest.TestCase):
    def test_keep_at_most_trims_excess_off_the_newest_end(self):
        # Columns are newest-first; any leaked-but-unannounced quarter is
        # always among the newest survivors, so keep_at_most must trim from
        # that end and keep the OLDEST N, not the newest N.
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-06-30", "2026-03-31", "2025-12-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-17", keep_at_most=1)
        self.assertEqual(list(out.columns), ["2025-12-31"])

    def test_keep_at_most_zero_drops_all_columns(self):
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-06-30", "2026-03-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-17", keep_at_most=0)
        self.assertEqual(list(out.columns), [])

    def test_none_keep_at_most_preserves_prior_behavior(self):
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-06-30", "2026-03-31", "2025-12-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-17", keep_at_most=None)
        self.assertEqual(list(out.columns), ["2026-06-30", "2026-03-31", "2025-12-31"])

    def test_date_mask_still_applies_before_the_cap(self):
        # A column strictly after curr_date is dropped by the existing date
        # mask regardless of keep_at_most.
        data = pd.DataFrame(
            {c: [1.0] for c in ["2026-09-30", "2026-06-30", "2026-03-31"]}
        )
        out = filter_financials_by_date(data, "2026-07-17", keep_at_most=5)
        self.assertEqual(list(out.columns), ["2026-06-30", "2026-03-31"])

    def test_empty_data_or_no_curr_date_unaffected(self):
        data = pd.DataFrame({"2026-06-30": [1.0]})
        self.assertTrue(filter_financials_by_date(data, None, keep_at_most=1).equals(data))
        empty = pd.DataFrame()
        self.assertTrue(filter_financials_by_date(empty, "2026-07-17", keep_at_most=1).empty)


@pytest.mark.unit
class AnnouncedQuartersHelperTests(unittest.TestCase):
    def _mock_ticker(self, earnings_df):
        ticker_obj = mock.Mock()
        ticker_obj.get_earnings_dates.return_value = earnings_df
        return ticker_obj

    def test_count_announced_quarters_excludes_future_and_unreported_rows(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        # curr_date is the day BEFORE the real 2026-07-18 announcement --
        # only the 3 older, already-reported rows should count.
        count = y_finance._count_announced_quarters(ticker_obj, "2026-07-17")
        self.assertEqual(count, 3)

    def test_count_announced_quarters_includes_same_day_announcement(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        count = y_finance._count_announced_quarters(ticker_obj, "2026-07-18")
        self.assertEqual(count, 4)

    def test_count_returns_none_on_fetch_failure(self):
        ticker_obj = mock.Mock()
        ticker_obj.get_earnings_dates.side_effect = RuntimeError("network down")
        self.assertIsNone(y_finance._count_announced_quarters(ticker_obj, "2026-07-17"))

    def test_unannounced_quarter_leaked_true_when_real_announcement_is_after_curr_date(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        self.assertTrue(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-17"))

    def test_unannounced_quarter_leaked_false_once_curr_date_is_on_or_after_it(self):
        ticker_obj = self._mock_ticker(_ICICIBANK_LIKE_EARNINGS)
        self.assertFalse(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-18"))

    def test_unannounced_quarter_leaked_false_on_fetch_failure(self):
        ticker_obj = mock.Mock()
        ticker_obj.get_earnings_dates.side_effect = RuntimeError("network down")
        self.assertFalse(y_finance._unannounced_quarter_leaked(ticker_obj, "2026-07-17"))


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
        # 2026-06-30's period end (< curr_date) would pass the naive date
        # mask, but its real announcement (2026-07-18) is after curr_date --
        # the fix must still exclude it. 3 quarters (Mar31/Dec31/Sep30) had
        # really been announced by curr_date per _ICICIBANK_LIKE_EARNINGS.
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

    def test_annual_frequency_is_not_capped_by_announced_quarter_count(self):
        # keep_at_most is quarterly-earnings-specific; annual statements keep
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


if __name__ == "__main__":
    unittest.main()
