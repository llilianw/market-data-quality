"""Configuration defaults, immutability and composition."""

from dataclasses import FrozenInstanceError
from datetime import time

import pytest

from market_quality.config import AnalyticsConfig, AppConfig, QualityConfig, SessionConfig


def test_defaults_match_declared_assumptions() -> None:
    config = AppConfig()
    assert config.session.timezone == "America/Chicago"
    assert config.session.session_start == time(17)
    assert config.session.session_end == time(16)
    assert config.analytics.vwap_window == "15min"
    assert config.analytics.daily_boundary == "session"
    assert config.quality.expected_frequency == "1min"
    assert config.quality.detect_outliers is False


@pytest.mark.parametrize(
    "config,attribute,value",
    [
        (SessionConfig(), "timezone", "UTC"),
        (AnalyticsConfig(), "vwap_window", "30min"),
        (QualityConfig(), "detect_outliers", True),
        (AppConfig(), "session", SessionConfig("UTC")),
    ],
)
def test_configuration_is_immutable(config, attribute, value) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(config, attribute, value)


def test_app_composes_custom_settings() -> None:
    session = SessionConfig("Europe/London", time(8), time(17))
    analytics = AnalyticsConfig(vwap_window="30min", daily_boundary="calendar")
    quality = QualityConfig("5min", True)
    config = AppConfig(session=session, analytics=analytics, quality=quality)
    assert config.session is session
    assert config.analytics is analytics
    assert config.quality is quality
    assert AppConfig().quality.detect_outliers is False
