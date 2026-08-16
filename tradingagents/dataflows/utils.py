import re
from datetime import date, datetime, timedelta, timezone
from typing import Annotated

import pandas as pd
from dateutil.relativedelta import relativedelta

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC-aware; a naive value is assumed to be UTC.

    Window bounds arrive naive (parsed from ``yyyy-mm-dd``) while article
    timestamps may be offset-aware, so every operand is normalized before
    comparison -- stripping tzinfo instead of converting (the old behavior)
    made filtering depend on the wall-clock offset the timestamp happened to
    carry, e.g. misreading 2025-05-10T01:00+05:00 (really 05-09T20:00Z) as
    05-10 and dropping it (#1126).
    """
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def in_news_window(pub_date, start_dt, end_dt) -> bool:
    """Whether an article belongs in the half-open window ``[start_dt, end_dt + 1 day)``.

    Shared by every news vendor (yfinance, Indian RSS, ...) so look-ahead
    safety is enforced identically regardless of source: dated articles are
    kept only if they fall in the window; an undated article is kept only
    when the window reaches the present (live run), since a historical/backtest
    window can't prove it isn't future news (#992/#1007). Every operand is
    normalized to UTC and the upper bound is exclusive, so an article stamped
    exactly at midnight after end_dt cannot leak into a historical run (#1126).
    """
    end = _as_utc(end_dt)
    if pub_date is not None:
        return _as_utc(start_dt) <= _as_utc(pub_date) < end + relativedelta(days=1)
    return end >= datetime.now(timezone.utc) - relativedelta(days=1)

# Tickers can contain letters, digits, dot, dash, underscore, caret
# (index symbols like ^GSPC), equals (futures like GC=F), plus
# (forex/CFD symbols like XAUUSD+), and ampersand (NSE symbols with '&' in
# the company name, e.g. M&M.NS -- Mahindra & Mahindra; TFR-215). None of
# these enable directory traversal, so the value never escapes a containing
# directory when interpolated into a path. Anything else is rejected.
_TICKER_PATH_RE = re.compile(r"^[A-Za-z0-9._\-\^=+&]+$")


def safe_ticker_component(value: str, *, max_len: int = 32) -> str:
    """Validate ``value`` is safe to interpolate into a filesystem path.

    Tickers come from user CLI input or from LLM tool calls, both of which
    can be influenced by attacker-controlled content (e.g. prompt injection
    embedded in fetched news). Without validation, a value like
    ``"../../../etc/foo"`` flows into ``os.path.join`` / ``Path /`` and
    escapes the configured cache, checkpoint, or results directory.

    Returns ``value`` unchanged when it matches the allowed pattern; raises
    ``ValueError`` otherwise.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"ticker must be a non-empty string, got {value!r}")
    if len(value) > max_len:
        raise ValueError(f"ticker exceeds {max_len} chars: {value!r}")
    if not _TICKER_PATH_RE.fullmatch(value):
        raise ValueError(
            f"ticker contains characters not allowed in a filesystem path: {value!r}"
        )
    # The regex above allows '.', so values like '.', '..', '...' would pass,
    # and as a path component they traverse the parent directory. Reject any
    # value that's only dots.
    if set(value) == {"."}:
        raise ValueError(f"ticker cannot consist solely of dots: {value!r}")
    return value


def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path, encoding="utf-8")
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date
