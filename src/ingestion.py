"""Stage 1 - data ingestion, data-quality register and the join to one clean analysis dataset.

Join logic (grain = one row per loan account)
---------------------------------------------
    loan_accounts  (14,685)                      <- base table, holds target_default
      LEFT JOIN customers          ON customer_id          many-to-one
      LEFT JOIN bureau_agg         ON customer_id          many-to-one   (bureau lines aggregated per customer first)
      LEFT JOIN repayment_agg      ON account_id           one-to-one    (12 monthly rows aggregated per account first)
      LEFT JOIN collections_agg    ON account_id           one-to-one    (events aggregated per account first)

Child tables are aggregated BEFORE joining so the base grain can never fan out; every merge uses
pandas' ``validate=`` and the final row count is asserted equal to the loan table.

Every data-quality finding is written to a *register* (what, how many, what we did, why) which the
dashboard shows and the README summarises. Nothing is silently dropped.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import bureau_features, collections_features, repayment_features

DATE_COLS = {
    "customers": ["customer_since"],
    "loan_accounts": ["origination_date"],
    "bureau_credit": ["date_opened", "date_closed"],
    "collections_events": ["event_date"],
}
NUM_COLS = {
    "customers": ["age", "years_employed", "annual_income", "num_dependents", "years_at_residence"],
    "loan_accounts": ["loan_amount", "interest_rate_pct", "tenure_months", "collateral_flag", "target_default"],
    "bureau_credit": ["sanctioned_amount", "outstanding_amount", "days_past_due"],
    "repayment_history": ["payment_month", "emi_due", "emi_paid", "days_past_due", "outstanding_balance"],
    "collections_events": ["amount_recovered"],
}
PKS = {
    "customers": ["customer_id"], "loan_accounts": ["account_id"], "bureau_credit": ["bureau_id"],
    "repayment_history": ["account_id", "payment_month"], "collections_events": ["collection_event_id"],
}


# --------------------------------------------------------------------------- register
class DQRegister:
    """Collects data-quality findings. Each row: table, check, affected, total, decision, rationale."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table: str, check: str, affected: int, total: int, decision: str, rationale: str,
            severity: str = "Medium") -> None:
        affected, total = int(affected), int(total)
        self.rows.append({
            "table": table, "check": check, "affected": affected, "total": total,
            "pct": round(100 * affected / total, 2) if total else 0.0,
            "status": "Pass" if affected == 0 else "Issue",
            "severity": "-" if affected == 0 else severity,
            "decision": decision if affected else "No action needed",
            "rationale": rationale,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


@dataclass
class Bundle:
    dataset: pd.DataFrame            # one row per loan account
    collections: pd.DataFrame        # one row per collections event, enriched with account dimensions
    dq: pd.DataFrame                 # data-quality register
    tables: dict                     # cleaned source tables (for row counts / drill-downs)


# --------------------------------------------------------------------------- loading
def _norm(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())).strip("_")


def find_file(table: str, raw_dir: Path = C.RAW_DIR) -> Path:
    by_norm = {_norm(p.stem): p for p in Path(raw_dir).glob("*.csv")}
    for cand in C.FILE_CANDIDATES[table]:
        if _norm(Path(cand).stem) in by_norm:
            return by_norm[_norm(Path(cand).stem)]
    raise FileNotFoundError(f"Missing '{table}' CSV in {raw_dir}. Accepted names: {C.FILE_CANDIDATES[table]}")


def load_raw(raw_dir: Path = C.RAW_DIR) -> dict[str, pd.DataFrame]:
    tables = {}
    for t in C.FILE_CANDIDATES:
        df = pd.read_csv(find_file(t, raw_dir), low_memory=False)
        df.columns = [_norm(c) for c in df.columns]
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
        tables[t] = df
    return tables


# --------------------------------------------------------------------------- cleaning + checks
def clean_tables(raw: dict[str, pd.DataFrame], dq: DQRegister) -> dict[str, pd.DataFrame]:
    t = {k: v.copy() for k, v in raw.items()}

    # 1. Primary-key duplicates --------------------------------------------------------------------
    for name, df in t.items():
        dup = df.duplicated(PKS[name], keep="first")
        dq.add(name, f"Duplicate primary key {PKS[name]}", dup.sum(), len(df),
               "Keep first occurrence, drop the rest", "A key must identify exactly one record.", "High")
        t[name] = df.loc[~dup].reset_index(drop=True)

    # 2. Dates: parse, count unparseable (incl. Excel '########') ----------------------------------
    for name, cols in DATE_COLS.items():
        for col in cols:
            s = t[name][col].replace({"########": np.nan, "": np.nan})
            parsed = pd.to_datetime(s, errors="coerce")
            bad = int((s.notna() & parsed.isna()).sum())
            dq.add(name, f"Unparseable date in {col}", bad, len(parsed), "Set to missing",
                   "Invalid dates cannot be used for time logic; count kept for audit.", "Medium")
            t[name][col] = parsed

    # 3. Numerics: coerce, count newly-missing ------------------------------------------------------
    for name, cols in NUM_COLS.items():
        for col in cols:
            before = t[name][col].isna().sum()
            t[name][col] = pd.to_numeric(t[name][col], errors="coerce").astype("float64")
            dq.add(name, f"Non-numeric value in {col}", t[name][col].isna().sum() - before, len(t[name]),
                   "Set to missing", "Text in a numeric field is a load error, not a value.", "Medium")

    cust, loans, bur, rep, col = (t[k] for k in ["customers", "loan_accounts", "bureau_credit",
                                                 "repayment_history", "collections_events"])

    # 4. Label hygiene: inconsistent gender coding (documented in the data dictionary) -------------
    g = cust["gender"].str.title()
    recode = {"M": "Male", "F": "Female"}
    n_recode = int(g.isin(recode).sum())
    cust["gender"] = g.replace(recode)
    dq.add("customers", "Inconsistent gender labels ('M'/'F' vs 'Male'/'Female')", n_recode, len(cust),
           "Recode M->Male, F->Female", "Same category written two ways would split one group in every cut.", "Medium")

    # 5. Referential integrity ----------------------------------------------------------------------
    o1 = ~loans["customer_id"].isin(cust["customer_id"])
    dq.add("loan_accounts", "Loan whose customer_id is not in customers", o1.sum(), len(loans),
           "Keep loan, customer attributes stay missing", "A loan is still real exposure.", "High")
    for name, df, key_df, key in [("bureau_credit", bur, cust, "customer_id"), ("repayment_history", rep, loans, "account_id"),
                                  ("collections_events", col, loans, "account_id")]:
        orphan = ~df[key].isin(key_df[key])
        dq.add(name, f"Row whose {key} has no parent", orphan.sum(), len(df),
               "Drop orphan child rows", "A child row that cannot be attributed to an account/customer cannot be used.", "High")
        t[name] = df.loc[~orphan].reset_index(drop=True)
    bur, rep, col = t["bureau_credit"], t["repayment_history"], t["collections_events"]

    # 5b. Coverage (not errors, but they drive modelling choices) ----------------------------------
    no_bureau = ~cust["customer_id"].isin(bur["customer_id"])
    dq.add("customers", "Customer with no bureau lines ('thin file')", no_bureau.sum(), len(cust),
           "Keep; counts/amounts = 0, bureau DPD left missing, explicit `bur_thin_file` flag",
           "No history is not the same as a clean history, so it must not be imputed as 'clean'.", "Medium")
    per_acc = rep.groupby("account_id").size()
    dq.add("repayment_history", "Account without exactly 12 monthly records", int((per_acc != 12).sum()) +
           int((~loans["account_id"].isin(per_acc.index)).sum()), len(loans),
           "Compute features on available months", "Features assume a comparable observation window.", "Medium")

    # 6. Missing values with explicit imputation decisions -----------------------------------------
    dq.add("customers", "years_employed missing", cust["years_employed"].isna().sum(), len(cust),
           "Median by employment_type + `years_employed_missing` flag",
           "Imputation is flagged so any information in the missingness itself is kept.", "Low")
    dq.add("customers", "num_dependents missing", cust["num_dependents"].isna().sum(), len(cust),
           "Median + `num_dependents_missing` flag",
           "Missing dependents carries signal (higher default rate) so the flag is a model feature.", "Low")
    dq.add("customers", "years_at_residence missing", cust["years_at_residence"].isna().sum(), len(cust),
           "Median + flag", "No evidence of non-random missingness.", "Low")
    dq.add("loan_accounts", "interest_rate_pct missing", loans["interest_rate_pct"].isna().sum(), len(loans),
           "Median by loan_type + `interest_rate_missing` flag",
           "Rates differ strongly by product (6.6% mortgage vs 19.5% card) so a global median would be wrong.", "Low")
    closed_missing = (bur["days_past_due"].isna() & (bur["account_status"] == "Closed"))
    all_closed_zero = bool((bur.loc[bur["account_status"] == "Closed", "days_past_due"].dropna() == 0).all())
    dq.add("bureau_credit", "days_past_due missing on a CLOSED line", closed_missing.sum(), len(bur),
           "Set to 0" if all_closed_zero else "Leave missing",
           "Every observed closed line has DPD = 0 (verified), so missing on a closed line is safely 0.", "Low")
    bur.loc[closed_missing & all_closed_zero, "days_past_due"] = 0.0
    dq.add("bureau_credit", "days_past_due missing on an ACTIVE/WRITTEN-OFF line", bur["days_past_due"].isna().sum(), len(bur),
           "Leave missing; aggregate over reported lines; count kept as `bur_dpd_missing_lines`",
           "Cannot infer delinquency of an open line; aggregates skip missing values.", "Medium")

    # 7. Plausibility / consistency ------------------------------------------------------------------
    dq.add("customers", f"Age outside {C.MIN_PLAUSIBLE_AGE}-{C.MAX_PLAUSIBLE_AGE}",
           (~cust["age"].between(C.MIN_PLAUSIBLE_AGE, C.MAX_PLAUSIBLE_AGE)).sum(), len(cust), "Flag", "-", "Medium")
    dq.add("customers", "years_employed > age - 16 (working before age 16)",
           (cust["years_employed"] > cust["age"] - C.MIN_WORKING_AGE).sum(), len(cust),
           "Cap at age - 16", "Biologically implausible; capped rather than dropped to keep the customer.", "Low")
    dq.add("customers", f"Income at the {C.INCOME_FLOOR:,} floor (censored)", (cust["annual_income"] <= C.INCOME_FLOOR).sum(), len(cust),
           "Keep; note censoring", "Floor looks like a data-entry minimum, mostly unemployed customers; true income may be lower.", "Low")
    dq.add("loan_accounts", "Non-positive loan_amount or tenure", ((loans["loan_amount"] <= 0) | (loans["tenure_months"] <= 0)).sum(),
           len(loans), "Drop", "Not a real loan.", "High")
    dq.add("loan_accounts", "target_default missing or not in {0,1}", (~loans["target_default"].isin([0, 1])).sum(), len(loans),
           "Exclude from supervised training; keep in exposure counts", "Labels are never imputed.", "High")
    dq.add("bureau_credit", "outstanding_amount > sanctioned_amount", (bur["outstanding_amount"] > bur["sanctioned_amount"]).sum(),
           len(bur), "Flag", "Over-limit balance may be real (fees) or an error.", "Low")
    dq.add("bureau_credit", "Closed line with outstanding balance > 0",
           ((bur["account_status"] == "Closed") & (bur["outstanding_amount"] > 0)).sum(), len(bur), "Flag", "Status/balance contradiction.", "Medium")
    dq.add("repayment_history", "emi_paid > emi_due", (rep["emi_paid"] > rep["emi_due"] + 0.01).sum(), len(rep), "Flag", "Overpayment.", "Low")
    inconsistent = ((rep["payment_status"] == "On-Time") & (rep["days_past_due"] > 0)) | \
                   ((rep["payment_status"] == "Missed") & (rep["emi_paid"] > 0))
    dq.add("repayment_history", "Status contradicts DPD / amount paid", inconsistent.sum(), len(rep), "Flag",
           "Status, DPD and payment must agree.", "Medium")
    partial = (rep["emi_paid"] > 0) & (rep["emi_paid"] < rep["emi_due"] - 0.01)
    dq.add("repayment_history", "Partial payments (0 < paid < due)", partial.sum(), len(rep), "n/a",
           "None exist: every month is either paid in full or fully missed, so a 'partial payment' feature would just duplicate 'missed'.", "Low")

    # 8. Temporal-logic problems (these shape what we may safely use) -------------------------------
    lc = loans.merge(cust[["customer_id", "customer_since"]], on="customer_id", how="left")
    dq.add("customers", "customer_since is AFTER the loan's origination_date", (lc["customer_since"] > lc["origination_date"]).sum(),
           len(lc), "Do not use customer_since as a feature / tenure",
           "A loan cannot pre-date the customer relationship; the field is unreliable for tenure.", "High")
    ev = col.merge(loans[["account_id", "origination_date"]], on="account_id", how="left")
    dq.add("collections_events", "Collections event dated BEFORE the loan was originated", (ev["event_date"] < ev["origination_date"]).sum(),
           len(ev), "Use only counts/amounts per account; no date-based collections logic; never a model input",
           "Event dates are unreliable, and collections happen AFTER delinquency, so they would leak the outcome.", "High")
    bm = bur.merge(loans[["customer_id", "origination_date"]], on="customer_id", how="inner")
    dq.add("bureau_credit", "Bureau line opened AFTER the customer's loan was originated", (bm["date_opened"] > bm["origination_date"]).sum(),
           len(bm), "Treat bureau as a CURRENT SNAPSHOT, not point-in-time; flag as production requirement",
           "A real origination model needs as-of-date bureau pulls; here it is behavioural / monitoring scoring.", "High")

    # 9. Apply the cleaning decisions above ----------------------------------------------------------
    cust["years_employed"] = np.minimum(cust["years_employed"], (cust["age"] - C.MIN_WORKING_AGE).clip(lower=0))
    t["customers"], t["loan_accounts"] = cust, loans[(loans["loan_amount"] > 0) & (loans["tenure_months"] > 0)].reset_index(drop=True)
    t["bureau_credit"] = bur
    return t


# --------------------------------------------------------------------------- dataset assembly
def _bands(df: pd.DataFrame) -> pd.DataFrame:
    df["age_band"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 200], labels=["<30", "30-39", "40-49", "50-59", "60+"], right=False).astype(str)
    df["income_band"] = pd.cut(df["annual_income"], [0, 40_000, 60_000, 80_000, 100_000, 1e12],
                               labels=["<40k", "40-60k", "60-80k", "80-100k", "100k+"], right=False).astype(str)
    df["origination_year"] = df["origination_date"].dt.year.astype("Int64").astype(str)
    df["origination_quarter"] = df["origination_date"].dt.to_period("Q").astype(str)
    df["secured"] = np.where(df["collateral_flag"] == 1, "Secured", "Unsecured")
    df["bureau_profile"] = np.select(
        [df["bur_thin_file"] == 1, df["bur_writeoff"] > 0], ["Thin file (no bureau lines)", "Has written-off line(s)"],
        default="No write-offs")
    return df


def build_dataset(raw_dir: Path = C.RAW_DIR, save: bool = True) -> Bundle:
    dq = DQRegister()
    raw = load_raw(raw_dir)
    t = clean_tables(raw, dq)
    cust, loans, bur, rep, col = (t[k] for k in ["customers", "loan_accounts", "bureau_credit", "repayment_history", "collections_events"])

    ba, ra, ca = bureau_features(bur), repayment_features(rep), collections_features(col)
    n0 = len(loans)
    df = (loans.merge(cust, on="customer_id", how="left", validate="many_to_one")
               .merge(ba, on="customer_id", how="left", validate="many_to_one")
               .merge(ra, on="account_id", how="left", validate="one_to_one")
               .merge(ca, on="account_id", how="left", validate="one_to_one"))
    assert len(df) == n0, "Join changed the grain - fan-out detected"          # the key safety check

    # Absence of events/lines is information, not missingness: counts and amounts become 0.
    for c in [c for c in df.columns if c.startswith("col_")]:
        df[c] = df[c].fillna(0)
    bur_zero = [c for c in df.columns if c.startswith("bur_") and c not in
                {"bur_max_dpd", "bur_avg_dpd", "bur_utilisation", "bur_writeoff_share", "bur_nbfc_fintech_share"}]
    df["bur_thin_file"] = df["bur_lines"].isna().astype(int)
    for c in bur_zero:
        df[c] = df[c].fillna(0)
    df["bur_writeoff_share"] = df["bur_writeoff_share"].fillna(0)
    df["bur_nbfc_fintech_share"] = df["bur_nbfc_fintech_share"].fillna(0)

    # Flagged imputations (decisions registered above).
    df["years_employed_missing"] = df["years_employed"].isna().astype(int)
    df["num_dependents_missing"] = df["num_dependents"].isna().astype(int)
    df["years_at_residence_missing"] = df["years_at_residence"].isna().astype(int)
    df["interest_rate_missing"] = df["interest_rate_pct"].isna().astype(int)
    df["years_employed"] = df["years_employed"].fillna(df.groupby("employment_type")["years_employed"].transform("median"))
    df["num_dependents"] = df["num_dependents"].fillna(df["num_dependents"].median())
    df["years_at_residence"] = df["years_at_residence"].fillna(df["years_at_residence"].median())
    df["interest_rate_pct"] = df["interest_rate_pct"].fillna(df.groupby("loan_type")["interest_rate_pct"].transform("median"))

    # Ratios used by both the model and the anomaly detector.
    inc = df["annual_income"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_amount"] / inc
    df["emi_to_income"] = df["rep_emi_avg"] * 12 / inc
    df["bur_outstanding_to_income"] = df["bur_outstanding"] / inc
    df["balance_paydown"] = 1 - df["rep_balance_end"] / df["loan_amount"]
    df["dpd_recent_vs_prior"] = df["rep_dpd_recent"] - df["rep_dpd_prior"]
    df = _bands(df)

    # Collections enriched with account dimensions (used by Ask the Portfolio's 'collections' dataset).
    dims = ["account_id", "loan_type", "region", "employment_type", "origination_year", "target_default"]
    colx = col.merge(df[dims], on="account_id", how="left")

    bundle = Bundle(df, colx, dq.frame(), t)
    if save:
        df.to_csv(C.PROC_DIR / C.ART["dataset"], index=False)
        colx.to_csv(C.PROC_DIR / C.ART["collections"], index=False)
        (C.PROC_DIR / C.ART["dq"]).write_text(json.dumps(dq.frame().to_dict("records"), indent=1))
    return bundle
