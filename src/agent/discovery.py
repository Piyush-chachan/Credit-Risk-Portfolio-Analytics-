"""Discover data sources and their real columns from loaded artifacts."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class SourceInfo:
    name: str
    description: str
    columns: list[str]
    row_count: int
    grain: str


def discover_sources(frames: dict[str, pd.DataFrame]) -> dict[str, SourceInfo]:
    grain_by_source = {
        "portfolio": "one row per loan account; customer, bureau and repayment features are pre-aggregated",
        "collections": "one row per collections event",
    }
    description = {
        "portfolio": "Analysis-ready account-level portfolio with outcome, borrower, bureau, repayment and score fields",
        "collections": "Collections event-level data joined to account dimensions",
    }
    result = {}
    for name, frame in frames.items():
        result[name] = SourceInfo(
            name=name,
            description=description.get(name, f"Loaded dataframe: {name}"),
            columns=list(frame.columns),
            row_count=len(frame),
            grain=grain_by_source.get(name, "grain must be confirmed from ingestion documentation"),
        )
    return result


def source_catalogue(frames: dict[str, pd.DataFrame]) -> str:
    lines = []
    for src in discover_sources(frames).values():
        lines.append(f"SOURCE {src.name}: {src.description}; grain={src.grain}; rows={src.row_count}")
        lines.append("columns: " + ", ".join(src.columns))
    return "\n".join(lines)
