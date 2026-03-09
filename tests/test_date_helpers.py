"""Tests for date/time helper functions."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from okb.mcp_server import parse_date_range, parse_since_filter
from okb.tools import format_relative_time, get_document_date


class TestGetDocumentDate:
    """Tests for get_document_date function."""

    def test_prefers_document_date(self):
        metadata = {"document_date": "2024-01-15", "file_modified_at": "2024-01-10"}
        assert get_document_date(metadata) == "2024-01-15"

    def test_falls_back_to_file_modified(self):
        metadata = {"file_modified_at": "2024-01-10"}
        assert get_document_date(metadata) == "2024-01-10"

    def test_returns_none_when_no_dates(self):
        metadata = {"title": "Test"}
        assert get_document_date(metadata) is None

    def test_returns_none_for_empty_metadata(self):
        assert get_document_date({}) is None


class TestFormatRelativeTime:
    """Tests for format_relative_time function."""

    @pytest.fixture(autouse=True)
    def freeze_time(self):
        """Freeze time to 2024-01-15 12:00:00 UTC for consistent tests."""
        frozen = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        with patch("okb.tools.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            yield frozen

    def test_just_now(self, freeze_time):
        # 30 seconds ago
        ts = (freeze_time - timedelta(seconds=30)).isoformat()
        assert format_relative_time(ts) == "just now"

    def test_minutes_ago(self, freeze_time):
        ts = (freeze_time - timedelta(minutes=5)).isoformat()
        assert format_relative_time(ts) == "5m ago"

    def test_hours_ago(self, freeze_time):
        ts = (freeze_time - timedelta(hours=3)).isoformat()
        assert format_relative_time(ts) == "3h ago"

    def test_days_ago(self, freeze_time):
        ts = (freeze_time - timedelta(days=5)).isoformat()
        assert format_relative_time(ts) == "5d ago"

    def test_months_ago(self, freeze_time):
        ts = (freeze_time - timedelta(days=45)).isoformat()
        assert format_relative_time(ts) == "1mo ago"

    def test_years_ago(self, freeze_time):
        ts = (freeze_time - timedelta(days=400)).isoformat()
        assert format_relative_time(ts) == "1y ago"

    def test_future_date(self, freeze_time):
        ts = (freeze_time + timedelta(days=5)).isoformat()
        assert format_relative_time(ts) == "future"

    def test_handles_z_suffix(self, freeze_time):
        ts = (freeze_time - timedelta(days=5)).isoformat().replace("+00:00", "Z")
        assert format_relative_time(ts) == "5d ago"

    def test_handles_date_only_string(self, freeze_time):
        # Date-only strings should work (naive datetime handling)
        assert format_relative_time("2024-01-10") == "5d ago"

    def test_invalid_timestamp_returns_empty(self):
        assert format_relative_time("not-a-date") == ""
        assert format_relative_time("") == ""


class TestParseSinceFilter:
    """Tests for parse_since_filter function."""

    @pytest.fixture(autouse=True)
    def freeze_time(self):
        """Freeze time for consistent tests."""
        frozen = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        with patch("okb.mcp_server.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            mock_dt.fromisoformat = datetime.fromisoformat
            yield frozen

    def test_days_filter(self, freeze_time):
        result = parse_since_filter("7d")
        expected = freeze_time - timedelta(days=7)
        assert result == expected

    def test_months_filter(self, freeze_time):
        result = parse_since_filter("6mo")
        expected = freeze_time - timedelta(days=180)
        assert result == expected

    def test_years_filter(self, freeze_time):
        result = parse_since_filter("1y")
        expected = freeze_time - timedelta(days=365)
        assert result == expected

    def test_case_insensitive(self, freeze_time):
        result1 = parse_since_filter("7D")
        result2 = parse_since_filter("7d")
        assert result1 == result2

    def test_iso_date(self):
        result = parse_since_filter("2024-01-01")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_iso_datetime_with_z(self):
        result = parse_since_filter("2024-01-01T12:00:00Z")
        assert result is not None
        assert result.hour == 12

    def test_invalid_returns_none(self):
        assert parse_since_filter("") is None
        assert parse_since_filter("invalid") is None

    def test_dateparser_fallback_last_week(self):
        """dateparser handles 'last week' — result should be in the past."""
        result = parse_since_filter("last week")
        assert result is not None
        assert result < datetime.now(UTC)

    def test_dateparser_fallback_3_months_ago(self):
        result = parse_since_filter("3 months ago")
        assert result is not None
        assert result < datetime.now(UTC) - timedelta(days=60)

    def test_dateparser_fallback_yesterday(self):
        result = parse_since_filter("yesterday")
        assert result is not None
        now = datetime.now(UTC)
        assert now - timedelta(days=2) < result < now


class TestParseDateRange:
    """Tests for parse_date_range function."""

    @pytest.fixture(autouse=True)
    def freeze_time(self):
        """Freeze time to Wednesday 2024-01-17 12:00:00 UTC."""
        frozen = datetime(2024, 1, 17, 12, 0, 0, tzinfo=UTC)
        with patch("okb.mcp_server.datetime") as mock_dt:
            mock_dt.now.return_value = frozen
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.strptime = datetime.strptime
            yield frozen

    def test_today(self, freeze_time):
        result = parse_date_range("today")
        assert result is not None
        start, end = result
        assert start.day == 17
        assert start.hour == 0
        assert end.day == 18
        assert end.hour == 0

    def test_tomorrow(self, freeze_time):
        result = parse_date_range("tomorrow")
        assert result is not None
        start, end = result
        assert start.day == 18
        assert end.day == 19

    def test_yesterday(self, freeze_time):
        result = parse_date_range("yesterday")
        assert result is not None
        start, end = result
        assert start.day == 16
        assert start.hour == 0
        assert end.day == 17
        assert end.hour == 0

    def test_this_week(self, freeze_time):
        # 2024-01-17 is Wednesday, so week starts Monday 2024-01-15
        result = parse_date_range("this_week")
        assert result is not None
        start, end = result
        assert start.day == 15  # Monday
        assert end.day == 22  # Following Monday

    def test_this_week_with_space(self, freeze_time):
        result = parse_date_range("this week")
        assert result is not None
        start, end = result
        assert start.day == 15
        assert end.day == 22

    def test_next_week(self, freeze_time):
        result = parse_date_range("next_week")
        assert result is not None
        start, end = result
        assert start.day == 22  # Next Monday
        assert end.day == 29  # Monday after

    def test_last_week(self, freeze_time):
        # 2024-01-17 is Wednesday, this week starts Mon 15, last week Mon 8-Sun 14
        result = parse_date_range("last_week")
        assert result is not None
        start, end = result
        assert start.day == 8  # Previous Monday
        assert end.day == 15  # This Monday

    def test_last_week_with_space(self, freeze_time):
        result = parse_date_range("last week")
        assert result is not None
        start, end = result
        assert start.day == 8
        assert end.day == 15

    def test_this_month(self, freeze_time):
        result = parse_date_range("this_month")
        assert result is not None
        start, end = result
        assert start.day == 1
        assert start.month == 1
        assert end.day == 1
        assert end.month == 2

    def test_this_month_with_space(self, freeze_time):
        result = parse_date_range("this month")
        assert result is not None
        start, end = result
        assert start.month == 1
        assert end.month == 2

    def test_next_month(self, freeze_time):
        result = parse_date_range("next_month")
        assert result is not None
        start, end = result
        assert start.day == 1
        assert start.month == 2
        assert end.day == 1
        assert end.month == 3

    def test_specific_date(self):
        result = parse_date_range("2024-02-15")
        assert result is not None
        start, end = result
        assert start.year == 2024
        assert start.month == 2
        assert start.day == 15
        assert end.day == 16

    def test_case_insensitive(self, freeze_time):
        result1 = parse_date_range("TODAY")
        result2 = parse_date_range("today")
        assert result1 == result2

    def test_invalid_date_format(self):
        assert parse_date_range("2024-13-01") is None  # Invalid month
        assert parse_date_range("") is None
        assert parse_date_range("not-a-date") is None

    def test_dateparser_fallback_natural_language(self):
        """dateparser handles 'january 15 2024' → single-day range."""
        result = parse_date_range("january 15 2024")
        assert result is not None
        start, end = result
        assert start.month == 1
        assert start.day == 15
        assert end.day == 16

    def test_dateparser_fallback_relative(self):
        """dateparser handles '2 days ago' → single-day range."""
        result = parse_date_range("2 days ago")
        assert result is not None
        start, end = result
        assert end - start == timedelta(days=1)
        assert start < datetime.now(UTC)
