# Acceptance Criteria

## Ingestion
- Support CSV and Parquet inputs.
- Normalise supported source schemas into one canonical schema.
- Handle malformed rows without failing an otherwise usable file.
- Fail clearly when required structural columns are missing.

## Session Handling
- Normalise timestamps explicitly using `America/Chicago`.
- Use a session-aware trading date by default.
- Exclude weekends and scheduled session breaks from gap defects.

## Data Quality
- Detect missing required values.
- Detect invalid OHLC relationships.
- Detect negative volumes.
- Treat zero volume as valid.
- Treat no-range bars as valid.
- Distinguish exact duplicates from conflicting duplicates.
- Detect missing timestamps during expected trading periods.

## Analytics
- Generate session-based daily OHLCV.
- Calculate rolling 15-minute VWAP using a time-based window.
- Use typical price `(H + L + C) / 3` as the fixed bar-price proxy.
- Return unavailable/NaN VWAP when rolling volume is zero.
- Never mix contracts or sessions in rolling calculations.

## Insights
- Identify recurring data-quality patterns.
- Provide evidence for generated insights.
- Suggest validation or cleansing rules.
- Core insight generation must work without an LLM.

## UI
- Allow contract and date-range filtering.
- Display market analytics and data-quality results.
- Show detailed quality issues and recurring insights.

## Testing
- Unit-test core domain rules and edge cases.
- Verify CSV/Parquet equivalence.
- Include at least one end-to-end backend integration test.
