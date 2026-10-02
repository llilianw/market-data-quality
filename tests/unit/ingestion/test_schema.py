import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import SchemaValidationError
from market_quality.ingestion.schema import resolve_market_fields


@pytest.fixture
def generic_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract": ["ESZ6"],
            "timestamp": ["not-yet-parsed"],
            "open": ["not-yet-coerced"],
            "high": [None],
            "low": [None],
            "close": [None],
            "volume": [-1],
        }
    )


def test_resolves_supplied_sample_schema(generic_data: pd.DataFrame) -> None:
    raw = generic_data.rename(
        columns={"contract": "contract_symbol", "timestamp": "timestamp_chicago_wall"}
    )
    assert resolve_market_fields(raw) == {
        "contract": "contract_symbol",
        "timestamp": "timestamp_chicago_wall",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }


def test_resolves_generic_schema_even_without_rows(generic_data: pd.DataFrame) -> None:
    assert resolve_market_fields(generic_data.iloc[:0]) == {
        "contract": "contract",
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }


@pytest.mark.parametrize("missing", [("volume",), ("contract", "timestamp")])
def test_reports_missing_logical_fields(
    generic_data: pd.DataFrame, missing: tuple[str, ...]
) -> None:
    with pytest.raises(SchemaValidationError) as caught:
        resolve_market_fields(generic_data.drop(columns=list(missing)))
    assert str(caught.value) == f"Missing required logical fields: {', '.join(missing)}"


def test_allows_unrelated_source_columns(generic_data: pd.DataFrame) -> None:
    raw = generic_data.assign(exchange="CME", timestamp_ms=123, source_note="original")
    assert resolve_market_fields(raw) == resolve_market_fields(generic_data)


def test_preserves_raw_dataframe(generic_data: pd.DataFrame) -> None:
    raw = generic_data.rename(
        columns={"contract": "contract_symbol", "timestamp": "timestamp_chicago_wall"}
    )
    before = raw.copy(deep=True)
    resolve_market_fields(raw)
    assert_frame_equal(raw, before)


@pytest.mark.parametrize(
    "field,other_column",
    [("contract", "contract_symbol"), ("timestamp", "timestamp_chicago_wall"), ("open", "open")],
)
def test_rejects_ambiguous_required_columns(
    generic_data: pd.DataFrame, field: str, other_column: str
) -> None:
    raw = pd.concat([generic_data, pd.DataFrame({other_column: [None]})], axis=1)
    with pytest.raises(SchemaValidationError) as caught:
        resolve_market_fields(raw)
    assert str(caught.value) == (
        f"Ambiguous source columns for logical fields: {field}: {field}, {other_column}"
    )
