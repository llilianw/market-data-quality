from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.ingestion.normalize import normalize_market_fields
from market_quality.ingestion.schema import resolve_market_fields


@pytest.fixture
def raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract": ["ESZ6", "NQZ6", "ESZ6"],
            "timestamp": ["2026-01-05 10:33:00", "not-a-timestamp", None],
            "open": ["100.0", "bad", None],
            "high": ["99.0", None, "10.0"],
            "low": ["101.0", "2.0", "10.0"],
            "close": ["100.0", "3.0", "10.0"],
            "volume": ["0", "-1", "bad"],
            "source_note": ["original", "malformed", "missing"],
        },
        index=[9, 2, 9],
    )


@pytest.mark.parametrize(
    "contract_column,timestamp_column,source_file",
    [
        ("contract_symbol", "timestamp_chicago_wall", Path("inputs/futures.csv")),
        ("contract", "timestamp", "inputs/../futures.csv"),
    ],
)
def test_normalizes_supported_schemas_with_positional_lineage(
    raw_data: pd.DataFrame, contract_column: str, timestamp_column: str, source_file: str | Path
) -> None:
    raw = raw_data.rename(columns={"contract": contract_column, "timestamp": timestamp_column})
    normalized = normalize_market_fields(raw, resolve_market_fields(raw), source_file)
    expected = pd.DataFrame(
        {
            "source_file": [str(source_file)] * 3,
            "source_row": [0, 1, 2],
            "contract": ["ESZ6", "NQZ6", "ESZ6"],
            "timestamp": ["2026-01-05 10:33:00", "not-a-timestamp", None],
            "open": ["100.0", "bad", None],
            "high": ["99.0", None, "10.0"],
            "low": ["101.0", "2.0", "10.0"],
            "close": ["100.0", "3.0", "10.0"],
            "volume": ["0", "-1", "bad"],
        }
    )
    assert_frame_equal(normalized, expected)


def test_uses_supplied_mapping_without_rediscovering_aliases(raw_data: pd.DataFrame) -> None:
    raw = raw_data.rename(columns={"contract": "ticker", "timestamp": "clock_text"})
    field_map = {
        "volume": "volume",
        "close": "close",
        "low": "low",
        "high": "high",
        "open": "open",
        "timestamp": "clock_text",
        "contract": "ticker",
    }
    normalized = normalize_market_fields(raw, field_map, "custom.csv")
    assert normalized.columns.tolist() == [
        "source_file",
        "source_row",
        "contract",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert normalized["contract"].tolist() == ["ESZ6", "NQZ6", "ESZ6"]
    assert_series_equal(
        normalized["timestamp"], raw["clock_text"].reset_index(drop=True).rename("timestamp")
    )


@pytest.mark.parametrize("label", ["exchange", "root"])
def test_rejects_duplicate_optional_metadata(raw_data: pd.DataFrame, label: str) -> None:
    metadata = pd.DataFrame(
        [["first", "second"]] * len(raw_data), columns=[label, label], index=raw_data.index
    )
    raw = pd.concat([raw_data, metadata], axis=1)
    before = raw.copy(deep=True)
    with pytest.raises(SchemaValidationError) as caught:
        normalize_market_fields(raw, resolve_market_fields(raw), "duplicates.csv")
    assert str(caught.value) == f"Duplicate optional metadata column labels: {label}"
    assert_frame_equal(raw, before)


def test_preserves_numeric_and_nullable_dtypes(raw_data: pd.DataFrame) -> None:
    raw = raw_data.assign(
        open=pd.array([100.0, float("nan"), 102.0], dtype="float32"),
        high=pd.array([101.0, pd.NA, 103.0], dtype="Float64"),
        volume=pd.array([0, pd.NA, -1], dtype="Int64"),
    )
    normalized = normalize_market_fields(raw, resolve_market_fields(raw), "numeric.csv")
    columns = ["open", "high", "volume"]
    assert_frame_equal(normalized[columns], raw[columns].reset_index(drop=True))


def test_preserves_optional_metadata_and_excludes_unrelated_columns(raw_data: pd.DataFrame) -> None:
    raw = raw_data.assign(exchange=["CME", None, "CME"], root=["ES", "NQ", "ES"])
    normalized = normalize_market_fields(raw, resolve_market_fields(raw), "metadata.csv")
    assert normalized.columns.tolist() == [
        "source_file",
        "source_row",
        "contract",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "exchange",
        "root",
    ]
    assert_frame_equal(
        normalized[["exchange", "root"]], raw[["exchange", "root"]].reset_index(drop=True)
    )


def test_does_not_mutate_input(raw_data: pd.DataFrame) -> None:
    before = raw_data.copy(deep=True)
    normalized = normalize_market_fields(raw_data, resolve_market_fields(raw_data), "source.csv")
    assert_frame_equal(raw_data, before)
    normalized.loc[0, "open"] = "changed"
    assert_frame_equal(raw_data, before)


def test_empty_source_retains_output_columns(raw_data: pd.DataFrame) -> None:
    raw = raw_data.iloc[:0]
    normalized = normalize_market_fields(raw, resolve_market_fields(raw), "empty.csv")
    assert normalized.empty
    assert normalized.columns.tolist() == [
        "source_file",
        "source_row",
        "contract",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
