"""Stable domain identifiers shared by later modules."""

from enum import StrEnum


class RuleId(StrEnum):
    """Stable identifiers for quality findings."""

    MISSING_REQUIRED_VALUE = "MISSING_REQUIRED_VALUE"
    INVALID_OHLC = "INVALID_OHLC"
    NEGATIVE_PRICE = "NEGATIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    CONFLICTING_DUPLICATE = "CONFLICTING_DUPLICATE"
    UNEXPECTED_GAP = "UNEXPECTED_GAP"
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"


class Severity(StrEnum):
    """Reporting severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class GapClassification(StrEnum):
    """Distinguish missing observations from declared expected closures."""

    UNEXPECTED_GAP = "unexpected_gap"
    EXPECTED_CLOSURE = "expected_closure"
