from pathlib import Path
from tempfile import NamedTemporaryFile

from market_quality.config import AppConfig
from market_quality.ingestion.normalize import (
    CanonicalizationResult,
    canonicalize_market_data,
    normalize_market_fields,
)
from market_quality.ingestion.readers import read_market_data
from market_quality.ingestion.schema import resolve_market_fields
from market_quality.pipeline import AnalysisResult, run_analysis


def process_uploaded_data(
    file_bytes: bytes,
    filename: str,
    config: AppConfig,
) -> tuple[CanonicalizationResult, AnalysisResult]:
    """Adapt uploaded bytes to the path-based reader, retaining original filename lineage."""
    with NamedTemporaryFile(suffix=Path(filename).suffix) as source:
        source.write(file_bytes)
        source.flush()
        raw = read_market_data(source.name)

    fields = resolve_market_fields(raw)
    normalized = normalize_market_fields(raw, fields, source_file=filename)
    canonical = canonicalize_market_data(normalized, config.session)
    assessment = run_analysis(canonical.data, config=config)
    return canonical, assessment
