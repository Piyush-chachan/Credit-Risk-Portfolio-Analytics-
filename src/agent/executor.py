"""Safe DuckDB execution. Identifiers are allow-listed; values are bound parameters."""
from __future__ import annotations
import duckdb
from src.ask_portfolio import build_sql


def execute_plan(connection: duckdb.DuckDBPyConnection, plan: dict):
    sql, params = build_sql(plan)
    return connection.execute(sql, params).df(), sql, params
