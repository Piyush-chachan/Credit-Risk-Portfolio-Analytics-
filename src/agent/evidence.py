"""Package query output with source lineage and execution details."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from src.ask_portfolio import DATASETS

@dataclass
class Evidence:
    dataset: str
    source: str
    source_rows: int
    result_rows: int
    plan: dict
    table: pd.DataFrame
    sql: str


def build_evidence(dataset: str, frame: pd.DataFrame, table: pd.DataFrame, plan: dict, sql: str) -> Evidence:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown source: {dataset}")
    return Evidence(dataset, DATASETS[dataset]["source"], len(frame), len(table), plan, table.copy(), sql)
