import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from market_quality.config import SessionConfig
from market_quality.exceptions import SchemaValidationError
from market_quality.ingestion.normalize import canonicalize_market_data


@pytest.fixture
def normalized() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": ["sample.csv"] * 3,
            "source_row": [0, 1, 2],
            "contract": ["ESZ6"] * 3,
            "timestamp": ["2026-01-05 10:31:00", "2026-01-05 10:32:00", "2026-01-05 10:33:00"],
            "open": ["100"] * 3,
            "high": ["101"] * 3,
            "low": ["99"] * 3,
            "close": ["100.5"] * 3,
            "volume": ["10"] * 3,
            "exchange": ["CME"] * 3,
            "root": ["ES"] * 3,
        }
    )


def test_chicago_wall_times_and_numeric_strings_become_canonical(normalized: pd.DataFrame) -> None:
    before = normalized.copy(deep=True)
    result = canonicalize_market_data(normalized)
    assert result.data.columns.tolist() == [
        "source_file",
        "source_row",
        "contract",
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "exchange",
        "root",
    ]
    assert str(result.data["timestamp_utc"].dt.tz) == "UTC"
    assert result.data.loc[0, "timestamp_utc"] == pd.Timestamp("2026-01-05T16:31:00Z")
    for field, value in {"open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10}.items():
        assert pd.api.types.is_numeric_dtype(result.data[field].dtype)
        assert result.data[field].tolist() == [value] * 3
    metadata = ["source_file", "source_row", "contract", "exchange", "root"]
    assert_frame_equal(result.data[metadata], normalized[metadata])
    assert result.issues.empty
    assert result.rejected_rows.empty
    assert_frame_equal(normalized, before)


@pytest.mark.parametrize("representation", ["typed", "text"])
def test_aware_timestamps_are_converted_without_relocalization(
    normalized: pd.DataFrame, representation: str
) -> None:
    expected = pd.Series(
        pd.to_datetime(
            ["2026-01-05T16:31:00Z", "2026-07-05T15:32:00Z", "2026-01-05T16:33:00Z"], utc=True
        ),
        name="timestamp_utc",
    )
    normalized["timestamp"] = (
        expected.dt.tz_convert("America/Chicago")
        if representation == "typed"
        else expected.astype("string")
    )
    result = canonicalize_market_data(normalized, SessionConfig(timezone="Pacific/Auckland"))
    assert_series_equal(result.data["timestamp_utc"], expected, check_dtype=False)
    assert str(result.data["timestamp_utc"].dt.tz) == "UTC"
    assert result.issues.empty


def test_naive_timestamps_use_configured_timezone(normalized: pd.DataFrame) -> None:
    result = canonicalize_market_data(normalized, SessionConfig(timezone="Europe/London"))
    assert result.data.loc[0, "timestamp_utc"] == pd.Timestamp("2026-01-05T10:31:00Z")


def test_parseable_market_defects_are_preserved_for_dq(normalized: pd.DataFrame) -> None:
    normalized = normalized.assign(
        open=["100", "-2", "50"],
        high=["100", "1", "49"],
        low=["100", "3", "51"],
        close=["100", "-1", "50"],
        volume=["0", "-500", "12"],
    )
    result = canonicalize_market_data(normalized)
    assert len(result.data) == 3
    assert result.data["volume"].tolist() == [0, -500, 12]
    assert result.data.loc[0, ["open", "high", "low", "close"]].tolist() == [100] * 4
    assert result.data["open"].tolist() == [100, -2, 50]
    assert result.data["high"].tolist() == [100, 1, 49]
    assert result.data["low"].tolist() == [100, 3, 51]
    assert result.issues.empty
    assert result.rejected_rows.empty


def test_existing_nulls_remain_canonical_without_ingestion_issues(normalized: pd.DataFrame) -> None:
    fields = ["timestamp", "open", "high", "low", "close", "volume"]
    normalized.loc[1, fields] = None
    result = canonicalize_market_data(normalized)
    assert len(result.data) == 3
    assert result.data.loc[1, ["timestamp_utc", *fields[1:]]].isna().all()
    assert str(result.data["timestamp_utc"].dt.tz) == "UTC"
    assert result.issues.empty
    assert result.rejected_rows.empty


@pytest.mark.parametrize("temporal", [pd.Timestamp("2026-01-05"), pd.Timedelta(days=1)])
def test_temporal_numeric_values_reject_only_non_null_rows(
    normalized: pd.DataFrame, temporal: pd.Timestamp | pd.Timedelta
) -> None:
    normalized = normalized.iloc[:2].copy()
    normalized["open"] = pd.Series([pd.NaT, temporal])
    before = normalized.copy(deep=True)
    result = canonicalize_market_data(normalized)
    assert result.data["source_row"].tolist() == [0]
    assert pd.isna(result.data.loc[0, "open"])
    assert pd.api.types.is_numeric_dtype(result.data["open"].dtype)
    assert len(result.issues) == 1
    issue = result.issues.iloc[0]
    assert (issue.source_file, issue.source_row, issue.field) == ("sample.csv", 1, "open")
    assert issue.issue_type == "MALFORMED_NUMERIC"
    assert issue.raw_value == temporal
    assert_frame_equal(result.rejected_rows, normalized.iloc[[1]].reset_index(drop=True))
    assert_frame_equal(normalized, before)


@pytest.mark.parametrize("token", ["now", "today"])
def test_relative_timestamp_tokens_are_rejected(normalized: pd.DataFrame, token: str) -> None:
    normalized.loc[1, "timestamp"] = token
    result = canonicalize_market_data(normalized)
    assert result.data["source_row"].tolist() == [0, 2]
    assert len(result.issues) == 1
    issue = result.issues.iloc[0]
    assert (issue.source_file, issue.source_row, issue.field) == ("sample.csv", 1, "timestamp")
    assert issue.issue_type == "TIMESTAMP_NORMALIZATION_FAILED"
    assert issue.raw_value == token
    assert_frame_equal(result.rejected_rows, normalized.iloc[[1]].reset_index(drop=True))


@pytest.mark.parametrize(
    "field,issue_type",
    [("timestamp", "TIMESTAMP_NORMALIZATION_FAILED"), ("volume", "MALFORMED_NUMERIC")],
)
def test_malformed_values_produce_issues_and_preserve_rejected_input(
    normalized: pd.DataFrame, field: str, issue_type: str
) -> None:
    normalized.loc[1, field] = "banana"
    before = normalized.copy(deep=True)
    result = canonicalize_market_data(normalized)
    assert result.data["source_row"].tolist() == [0, 2]
    assert result.issues.columns.tolist() == [
        "source_file",
        "source_row",
        "field",
        "issue_type",
        "raw_value",
        "message",
    ]
    issue = result.issues.iloc[0]
    assert len(result.issues) == 1
    assert (issue.source_file, issue.source_row, issue.field) == ("sample.csv", 1, field)
    assert issue.issue_type == issue_type
    assert issue.raw_value == "banana"
    assert "cannot" in issue.message.lower()
    assert_frame_equal(result.rejected_rows, normalized.iloc[[1]].reset_index(drop=True))
    assert_frame_equal(normalized, before)


def test_multiple_malformed_fields_reject_a_row_only_once(normalized: pd.DataFrame) -> None:
    normalized.loc[1, ["timestamp", "open", "volume"]] = "banana"
    result = canonicalize_market_data(normalized)
    assert set(result.issues["field"]) == {"timestamp", "open", "volume"}
    assert result.issues["source_row"].tolist() == [1, 1, 1]
    assert_frame_equal(result.rejected_rows, normalized.iloc[[1]].reset_index(drop=True))
    assert result.data["source_row"].tolist() == [0, 2]


def test_ambiguous_and_nonexistent_chicago_dst_times_are_rejected(normalized: pd.DataFrame) -> None:
    normalized["timestamp"] = ["2026-11-01 01:30:00", "2026-03-08 02:30:00", "2026-11-01 03:30:00"]
    result = canonicalize_market_data(normalized)
    assert result.data["source_row"].tolist() == [2]
    assert result.data.loc[0, "timestamp_utc"] == pd.Timestamp("2026-11-01T09:30:00Z")
    assert result.issues["source_row"].tolist() == [0, 1]
    assert set(result.issues["issue_type"]) == {"TIMESTAMP_NORMALIZATION_FAILED"}
    assert_frame_equal(result.rejected_rows, normalized.iloc[:2].reset_index(drop=True))


def test_empty_input_has_typed_utc_data_and_empty_diagnostics(normalized: pd.DataFrame) -> None:
    raw = normalized.drop(columns=["exchange", "root"]).iloc[:0]
    result = canonicalize_market_data(raw)
    assert result.data.empty
    assert str(result.data["timestamp_utc"].dt.tz) == "UTC"
    for field in ("open", "high", "low", "close", "volume"):
        assert pd.api.types.is_numeric_dtype(result.data[field].dtype)
    assert "exchange" not in result.data
    assert "root" not in result.data
    assert "row_id" not in result.data
    assert result.issues.empty
    assert result.issues.columns.tolist() == [
        "source_file",
        "source_row",
        "field",
        "issue_type",
        "raw_value",
        "message",
    ]
    assert_frame_equal(result.rejected_rows, raw.reset_index(drop=True))


def test_numeric_epoch_timestamp_requires_explicit_units(normalized: pd.DataFrame) -> None:
    normalized["timestamp"] = normalized["timestamp"].astype(object)
    normalized.loc[1, "timestamp"] = 1767607200000
    result = canonicalize_market_data(normalized)
    assert result.data["source_row"].tolist() == [0, 2]
    assert result.issues.loc[0, "raw_value"] == 1767607200000
    assert result.issues.loc[0, "field"] == "timestamp"


def test_mixed_timezone_representations_fail_clearly(normalized: pd.DataFrame) -> None:
    normalized.loc[1, "timestamp"] = "2026-01-05T10:32:00-06:00"
    with pytest.raises(SchemaValidationError, match="homogeneous timezone representation"):
        canonicalize_market_data(normalized)
