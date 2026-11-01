"""LLM query-plan generation. The model proposes JSON, never SQL or Python."""
from __future__ import annotations
import json

SYSTEM = """You are a query planner for a credit-risk portfolio analytics application.
Return one JSON object only. Never answer the analyst and never write SQL or code.
The plan must use only the supplied dataset, metric, dimension and category catalogue.
Schema: {status: ok|out_of_scope|unanswerable, reason?: string, dataset: string,
metrics: [metric names], group_by: [dimension names], filters: {dimension: [exact values]},
is_null: [dimension names], not_null: [dimension names], sort_by: metric name,
order: asc|desc, limit: integer|null}.
Use is_null for NULL values and not_null for IS NOT NULL. Do not invent categories.
Decline forecasts, causal explanations, external benchmarks, personal identifiers, and unsupported fields.
"""


def create_plan(llm, question: str, catalogue: str, history: list[dict] | None = None) -> dict:
    if llm is None:
        raise RuntimeError("No LLM configured")
    user = json.dumps({"question": question, "previous_turns": (history or [])[-3:], "catalogue": catalogue}, default=str)
    plan = llm.chat_json(SYSTEM, user, max_tokens=700)
    if not isinstance(plan, dict):
        raise ValueError("Planner did not return a JSON object")
    return plan
