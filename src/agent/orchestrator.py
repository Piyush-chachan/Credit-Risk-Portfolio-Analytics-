"""Thin modular orchestration helpers used by tests and future extensions.

The production Streamlit app currently calls PortfolioAssistant, which remains the
stable public facade. These helpers expose the individual retrieval stages without
creating a second SQL implementation.
"""
from __future__ import annotations
from src.agent.intent import detect_intent
from src.agent.discovery import source_catalogue
from src.agent.planner import create_plan
from src.agent.validator import validate_plan
from src.agent.executor import execute_plan
from src.agent.evidence import build_evidence
from src.agent.response import compose_answer


def run_retrieval(question, frames, connection, llm=None, history=None):
    intent = detect_intent(question)
    if intent.kind != "portfolio_question":
        return {"status": "declined", "reason": intent.reason, "stage": "intent"}
    catalogue = source_catalogue(frames)
    if llm is None:
        return {"status": "needs_fallback", "reason": "No LLM configured; call the existing rule-based planner."}
    raw_plan = create_plan(llm, question, catalogue, history)
    if raw_plan.get("status") != "ok":
        return {"status": "declined", "reason": raw_plan.get("reason", "Unsupported question"), "stage": "planning"}
    plan = validate_plan(raw_plan, frames)
    table, sql, params = execute_plan(connection, plan)
    if table.empty:
        return {"status": "declined", "reason": "No records match those filters.", "stage": "execution"}
    answer, warnings = compose_answer(question, plan, table)
    evidence = build_evidence(plan["dataset"], frames[plan["dataset"]], table, plan, sql)
    return {"status": "answered", "answer": answer, "warnings": warnings, "plan": plan,
            "evidence": evidence, "sql": sql, "params": params}
