"""Stable domain vocabulary and minimal exception hierarchy."""

from market_quality.constants import CANONICAL_COLUMNS, OHLC_COLUMNS
from market_quality.exceptions import (
    MarketDataError,
    SchemaValidationError,
    UnsupportedFileTypeError,
)
from market_quality.models import GapClassification, RuleId, Severity


def test_domain_identifiers_are_stable() -> None:
    assert {rule.name: rule.value for rule in RuleId} == {
        "MISSING_REQUIRED_VALUE": "MISSING_REQUIRED_VALUE",
        "INVALID_OHLC": "INVALID_OHLC",
        "NEGATIVE_PRICE": "NEGATIVE_PRICE",
        "NEGATIVE_VOLUME": "NEGATIVE_VOLUME",
        "EXACT_DUPLICATE": "EXACT_DUPLICATE",
        "CONFLICTING_DUPLICATE": "CONFLICTING_DUPLICATE",
        "UNEXPECTED_GAP": "UNEXPECTED_GAP",
        "STATISTICAL_OUTLIER": "STATISTICAL_OUTLIER",
    }
    assert {severity.name: severity.value for severity in Severity} == {
        "INFO": "info",
        "WARNING": "warning",
        "ERROR": "error",
    }
    assert {classification.name: classification.value for classification in GapClassification} == {
        "UNEXPECTED_GAP": "unexpected_gap",
        "EXPECTED_CLOSURE": "expected_closure",
    }


def test_shared_column_vocabulary() -> None:
    assert OHLC_COLUMNS == ("open", "high", "low", "close")
    assert CANONICAL_COLUMNS == (
        "row_id",
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
    )


def test_exception_hierarchy() -> None:
    assert issubclass(MarketDataError, Exception)
    assert issubclass(UnsupportedFileTypeError, MarketDataError)
    assert issubclass(SchemaValidationError, MarketDataError)
