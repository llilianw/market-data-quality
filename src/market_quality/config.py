"""Immutable configuration defaults shared by processing stages."""

from dataclasses import dataclass, field
from datetime import time
from typing import Literal


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Session boundaries are local wall-clock times interpreted in `timezone`."""

    timezone: str = "America/Chicago"
    session_start: time = time(17, 0)
    session_end: time = time(16, 0)


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Daily boundary and approximate bar-VWAP settings."""

    vwap_window: str = "15min"
    daily_boundary: Literal["session", "calendar"] = "session"


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Expected observation cadence and optional outlier detection."""

    expected_frequency: str = "1min"
    detect_outliers: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Typed application defaults; validation is owned by consuming stages."""

    session: SessionConfig = field(default_factory=SessionConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
