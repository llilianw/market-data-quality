"""Group-level duplicate findings without choosing surviving observations."""

import pandas as pd

from market_quality.constants import OHLC_COLUMNS
from market_quality.exceptions import SchemaValidationError
from market_quality.models import RuleId, Severity
from market_quality.quality.constants import ISSUE_COLUMNS


def detect_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Compare OHLCV for complete contract/timestamp keys, ignoring metadata.

    Matching missing market values compare equal; missing keys are not grouped.
    One disagreement makes every member of a key group conflicting. Evidence
    uses field='ohlcv' with a dictionary of the participant's market values.
    """
    key = ["contract", "timestamp_utc"]
    market_fields = [*OHLC_COLUMNS, "volume"]
    missing = [field for field in [*key, *market_fields] if field not in data.columns]
    if missing:
        raise SchemaValidationError(f"Missing required canonical columns: {', '.join(missing)}")

    complete_key = data.loc[:, key].notna().all(axis=1)
    participants = data.loc[complete_key & data.duplicated(key, keep=False)].reset_index(drop=True)
    if participants.empty:
        return pd.DataFrame(columns=ISSUE_COLUMNS).astype(
            {"blocking": bool, "timestamp_utc": data["timestamp_utc"].dtype}
        )

    groups = participants.groupby(key, sort=False, observed=True)
    group_size = groups["contract"].transform("size")
    # Counting null as a value distinguishes missing/populated without guessing a fill sentinel.
    conflicting = groups[market_fields].transform("nunique", dropna=False).gt(1).any(axis=1)
    issues = participants.reindex(
        columns=["source_file", "source_row", "contract", "timestamp_utc"]
    ).copy()
    issues["rule_id"] = RuleId.EXACT_DUPLICATE.value
    issues.loc[conflicting, "rule_id"] = RuleId.CONFLICTING_DUPLICATE.value
    issues["severity"] = Severity.WARNING.value
    issues.loc[conflicting, "severity"] = Severity.ERROR.value
    issues["blocking"] = conflicting
    issues["field"] = "ohlcv"
    issues["actual_value"] = participants.loc[:, market_fields].to_dict("records")
    description = pd.Series("Exact duplicate", index=participants.index)
    description.loc[conflicting] = "Conflicting duplicate"
    issues["message"] = (
        description
        + ": "
        + group_size.astype(str)
        + " observations share ("
        + participants["contract"].astype(str)
        + ", "
        + participants["timestamp_utc"].astype(str)
        + ")"
    )
    return issues.loc[:, ISSUE_COLUMNS]
