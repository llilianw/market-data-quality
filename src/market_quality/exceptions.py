"""Controlled errors exposed by the market-data domain."""


class MarketDataError(Exception):
    """Base class for market-data processing errors."""


class UnsupportedFileTypeError(MarketDataError):
    """An input file type is not supported."""


class SchemaValidationError(MarketDataError):
    """An input does not satisfy the required structural schema."""
