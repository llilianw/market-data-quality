from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_quality.exceptions import UnsupportedFileTypeError
from market_quality.ingestion.readers import read_market_data

CSV_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "market_data.csv"


@pytest.fixture
def raw_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_symbol": ["ESZ6", "ESZ6", "NQZ6"],
            "timestamp_chicago_wall": [
                "2026-01-05 10:31:00",
                "2026-01-05 10:32:00",
                "2026-01-05 10:33:00",
            ],
            "open": [100.0, 100.0, 90.0],
            "high": [101.0, 100.0, 89.0],
            "low": [99.0, 100.0, 91.0],
            "close": [100.5, 100.0, 90.0],
            "volume": [10, 0, -1],
            "source_note": ["original", "no_range", "unvalidated"],
        }
    )


@pytest.fixture
def parquet_path(tmp_path: Path, raw_data: pd.DataFrame) -> Path:
    path = tmp_path / "market_data.parquet"
    raw_data.to_parquet(path, engine="pyarrow", index=False)
    return path


def test_csv_preserves_source_data(raw_data: pd.DataFrame) -> None:
    assert_frame_equal(read_market_data(str(CSV_PATH)), raw_data)


def test_parquet_preserves_source_data(parquet_path: Path, raw_data: pd.DataFrame) -> None:
    assert_frame_equal(read_market_data(parquet_path), raw_data)


def test_csv_and_parquet_are_equivalent(parquet_path: Path) -> None:
    assert_frame_equal(read_market_data(CSV_PATH), read_market_data(parquet_path))


def test_unsupported_extension_raises_domain_error(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type '\.txt'"):
        read_market_data(tmp_path / "market_data.txt")
