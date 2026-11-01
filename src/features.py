"""Feature engineering: collapse the three one-to-many tables to analysis grain.

Grains
------
* repayment_history  -> one row per ACCOUNT   (12 monthly records each)
* bureau_credit      -> one row per CUSTOMER  (a customer-level snapshot, shared by all of the customer's loans)
* collections_events -> one row per ACCOUNT   (analytics / anomaly review only - never a model input)

All functions are vectorised (no row loops) and return plain float64/int64 columns - no pandas
nullable dtypes - so downstream numpy code never meets pd.NA.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EARLY_MONTHS = 6        # "early window" used for the leakage-robustness model
RECENT_MONTHS = 3       # "recent" window used for own-history deterioration


def repayment_features(rep: pd.DataFrame) -> pd.DataFrame:
    r = rep.sort_values(["account_id", "payment_month"]).copy()
    r["late"] = (r["payment_status"] == "Late").astype(int)
    r["missed"] = (r["payment_status"] == "Missed").astype(int)
    r["delq"] = ((r["late"] + r["missed"]) > 0).astype(int)
    n_months = int(r["payment_month"].max())
    recent_start = n_months - RECENT_MONTHS + 1

    g = r.groupby("account_id")
    out = g.agg(
        rep_months=("payment_month", "count"),
        rep_late=("late", "sum"),
        rep_missed=("missed", "sum"),
        rep_delinquent=("delq", "sum"),
        rep_avg_dpd=("days_past_due", "mean"),
        rep_max_dpd=("days_past_due", "max"),
        rep_emi_avg=("emi_due", "mean"),
        rep_paid_total=("emi_paid", "sum"),
        rep_due_total=("emi_due", "sum"),
    )
    early = r[r["payment_month"] <= EARLY_MONTHS].groupby("account_id")
    out["rep_delinquent_early"] = early["delq"].sum()
    out["rep_missed_early"] = early["missed"].sum()
    out["rep_avg_dpd_early"] = early["days_past_due"].mean()
    out["rep_delinquent_late_half"] = out["rep_delinquent"] - out["rep_delinquent_early"]

    recent = r[r["payment_month"] >= recent_start].groupby("account_id")
    prior = r[r["payment_month"] < recent_start].groupby("account_id")
    out["rep_dpd_recent"] = recent["days_past_due"].mean()          # own-history comparison: last 3 months ...
    out["rep_dpd_prior"] = prior["days_past_due"].mean()            # ... versus the account's previous 9 months
    out["rep_dpd_prior_std"] = prior["days_past_due"].std()
    out["rep_delinquent_recent"] = recent["delq"].sum()
    out["rep_delinquent_prior"] = prior["delq"].sum()
    out["rep_missed_recent"] = recent["missed"].sum()

    # Longest consecutive run of delinquent months (vectorised run-length trick).
    r["_run"] = (1 - r["delq"]).groupby(r["account_id"]).cumsum()
    runs = r[r["delq"] == 1].groupby(["account_id", "_run"]).size()
    out["rep_longest_streak"] = runs.groupby("account_id").max()
    out["rep_longest_streak"] = out["rep_longest_streak"].fillna(0)

    out["rep_balance_end"] = r.groupby("account_id")["outstanding_balance"].last()
    out["rep_pay_ratio"] = out["rep_paid_total"] / out["rep_due_total"].replace(0, np.nan)
    return out.reset_index()


def bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    b = bureau.copy()
    b["active"] = (b["account_status"] == "Active").astype(int)
    b["closed"] = (b["account_status"] == "Closed").astype(int)
    b["writeoff"] = (b["account_status"] == "Written-Off").astype(int)
    b["wo_outstanding"] = b["outstanding_amount"] * b["writeoff"]
    b["nbfc_fintech"] = b["lender_category"].isin(["NBFC", "Fintech"]).astype(int)
    b["dpd_missing"] = b["days_past_due"].isna().astype(int)
    b["live_sanctioned"] = b["sanctioned_amount"] * (1 - b["closed"])     # exposure that is not closed
    g = b.groupby("customer_id")
    out = g.agg(
        bur_lines=("bureau_id", "count"),
        bur_active=("active", "sum"),
        bur_closed=("closed", "sum"),
        bur_writeoff=("writeoff", "sum"),
        bur_sanctioned=("sanctioned_amount", "sum"),
        bur_outstanding=("outstanding_amount", "sum"),
        bur_wo_outstanding=("wo_outstanding", "sum"),
        bur_live_sanctioned=("live_sanctioned", "sum"),
        bur_max_dpd=("days_past_due", "max"),
        bur_avg_dpd=("days_past_due", "mean"),
        bur_dpd_missing_lines=("dpd_missing", "sum"),
        bur_nbfc_fintech=("nbfc_fintech", "sum"),
    ).reset_index()
    out["bur_utilisation"] = out["bur_outstanding"] / out["bur_sanctioned"].replace(0, np.nan)
    out["bur_writeoff_share"] = out["bur_writeoff"] / out["bur_lines"]
    out["bur_nbfc_fintech_share"] = out["bur_nbfc_fintech"] / out["bur_lines"]
    return out


def collections_features(col: pd.DataFrame) -> pd.DataFrame:
    c = col.copy()
    c["legal"] = (c["action_type"] == "Legal Notice").astype(int)
    c["field"] = (c["action_type"] == "Field Visit").astype(int)
    c["recovered_event"] = (c["outcome"] == "Payment Received").astype(int)
    return c.groupby("account_id").agg(
        col_events=("collection_event_id", "count"),
        col_recovered=("amount_recovered", "sum"),
        col_legal_notices=("legal", "sum"),
        col_field_visits=("field", "sum"),
        col_payments_received=("recovered_event", "sum"),
    ).reset_index()
