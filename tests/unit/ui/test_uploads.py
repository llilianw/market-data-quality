from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from market_quality.config import AppConfig, SessionConfig
from market_quality.exceptions import SchemaValidationError, UnsupportedFileTypeError
from market_quality.models import RuleId
from market_quality.ui.uploads import process_uploaded_data

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "market_data.csv"


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_upload_adapter_runs_existing_ingestion_and_full_assessment(extension: str) -> None:
    contents = _FIXTURE.read_bytes()
    if extension == "parquet":
        buffer = BytesIO()
        pd.read_csv(BytesIO(contents)).to_parquet(buffer, index=False)
        contents = buffer.getvalue()
    filename = f"uploaded.{extension}"
    config = AppConfig(session=SessionConfig(timezone="Europe/London"))
    canonical, assessment = process_uploaded_data(contents, filename, config)

    assert len(canonical.data) == 3
    assert canonical.issues.empty
    assert canonical.rejected_rows.empty
    assert canonical.data["source_file"].tolist() == [filename] * 3
    assert canonical.data["source_row"].tolist() == [0, 1, 2]
    assert canonical.data.loc[0, "timestamp_utc"] == pd.Timestamp("2026-01-05T10:31:00Z")
    assert str(assessment.enriched_data["local_timestamp"].dt.tz) == "Europe/London"
    assert set(assessment.quality_issues["rule_id"]) == {
        RuleId.INVALID_OHLC.value,
        RuleId.NEGATIVE_VOLUME.value,
    }
    assert assessment.eligible_data["source_row"].tolist() == [0, 1]
    pd.testing.assert_frame_equal(assessment.scoped_data, assessment.eligible_data)


def test_upload_preserves_ingestion_issues_and_original_rejected_values() -> None:
    raw = pd.read_csv(_FIXTURE).iloc[:2].copy()
    raw["volume"] = raw["volume"].astype(object)
    raw.loc[0, "volume"] = "bad-volume"
    canonical, assessment = process_uploaded_data(
        raw.to_csv(index=False).encode(), "bars.csv", AppConfig()
    )

    assert canonical.data["source_row"].tolist() == [1]
    assert canonical.rejected_rows["source_row"].tolist() == [0]
    assert canonical.rejected_rows.loc[0, "volume"] == "bad-volume"
    assert canonical.issues["issue_type"].tolist() == ["MALFORMED_NUMERIC"]
    assert canonical.issues["source_file"].tolist() == ["bars.csv"]
    assert assessment.enriched_data["source_row"].tolist() == [1]


@pytest.mark.parametrize("rejected", [False, True])
def test_empty_or_fully_rejected_upload_still_returns_assessment(rejected: bool) -> None:
    raw = pd.read_csv(_FIXTURE).iloc[: 1 if rejected else 0].copy()
    if rejected:
        raw["timestamp_chicago_wall"] = "not-a-timestamp"
    canonical, assessment = process_uploaded_data(
        raw.to_csv(index=False).encode(), "bars.csv", AppConfig()
    )

    assert canonical.data.empty
    assert len(canonical.rejected_rows) == int(rejected)
    assert len(canonical.issues) == int(rejected)
    assert assessment.enriched_data.empty
    assert assessment.eligible_data.empty
    assert assessment.daily_ohlcv.empty
    assert assessment.rolling_vwap.empty


@pytest.mark.parametrize(
    ("contents", "filename", "error"),
    [
        (b"contract\nESZ6\n", "incomplete.csv", SchemaValidationError),
        (b"unsupported", "bars.txt", UnsupportedFileTypeError),
    ],
)
def test_expected_domain_errors_propagate_to_the_app(
    contents: bytes, filename: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        process_uploaded_data(contents, filename, AppConfig())


def test_upload_cache_reuse_invalidation_and_copy_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    import app

    process = Mock(wraps=process_uploaded_data)
    monkeypatch.setattr(app, "process_uploaded_data", process)
    contents = (
        b"contract,timestamp,open,high,low,close,volume\n"
        b"ESZ6,2026-01-05 10:31:00,100,101,99,100.5,10\n"
    )
    app.assess_upload.clear()
    try:
        _, assessment = app.assess_upload(contents, "bars.csv", AppConfig())
        assert process.call_count == 1
        original = assessment.enriched_data.copy(deep=True)
        assessment.enriched_data.loc[0, "close"] = -999

        _, cached = app.assess_upload(contents, "bars.csv", AppConfig())
        assert process.call_count == 1
        pd.testing.assert_frame_equal(cached.enriched_data, original)

        renamed, _ = app.assess_upload(contents, "renamed.csv", AppConfig())
        assert process.call_count == 2
        assert renamed.data["source_file"].tolist() == ["renamed.csv"]

        changed_contents = contents.replace(b"100.5,10\n", b"100.5,20\n")
        changed, _ = app.assess_upload(changed_contents, "renamed.csv", AppConfig())
        assert process.call_count == 3
        assert changed.data["volume"].tolist() == [20]

        config = AppConfig(session=SessionConfig(timezone="Europe/London"))
        reconfigured, _ = app.assess_upload(changed_contents, "renamed.csv", config)
        assert process.call_count == 4
        assert reconfigured.data.loc[0, "timestamp_utc"] == pd.Timestamp("2026-01-05T10:31:00Z")
    finally:
        app.assess_upload.clear()
