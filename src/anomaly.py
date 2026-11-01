"""AI feature 2 - anomaly detection with an explainable review queue.

The brief asks for accounts with unusual repayment or bureau patterns *compared with their own history
or their peer segment*. Four independent detectors are combined; every flag carries plain-English
reasons with the numbers behind them, so an analyst can accept or dismiss it in seconds.

  A. OWN-HISTORY   the account's last 3 months versus its own previous 9 months
                   (sudden deterioration, new delinquency after a clean record, sustained streak)
  B. PEER OUTLIER  a metric in the top 2% of the account's peer segment (loan type x income tier)
  C. BUREAU vs INTERNAL DIVERGENCE
                   stressed at other lenders while paying us on time (early warning), or
                   missing payments to us while clean elsewhere (account-specific problem / dispute)
  D. MULTIVARIATE  IsolationForest fitted *within* each peer segment (combinations no single rule sees)

The target label (target_default) is never used. It is only used afterwards to measure lift.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import config as C

PEER_PCTL = 0.98            # top 2% of the peer segment
MIN_PEER_N = 100            # smaller segments fall back to loan_type only
IF_TAIL = 0.03              # bottom 3% isolation scores within a segment
QUEUE_MIN_SCORE = 3.0

PEER_METRICS = {            # column: (label, absolute floor so near-zero values are never 'outliers', formatter)
    "rep_avg_dpd": ("average repayment DPD", 12.0, "{:.1f} days"),
    "rep_missed": ("missed payments (12m)", 3.0, "{:.0f}"),
    "rep_max_dpd": ("worst repayment DPD", 60.0, "{:.0f} days"),
    "bur_avg_dpd": ("average bureau DPD", 30.0, "{:.1f} days"),
    "bur_utilisation": ("bureau utilisation", 0.60, "{:.0%}"),
    "emi_to_income": ("EMI-to-income", 0.40, "{:.0%}"),
    "loan_to_income": ("loan-to-income", 2.0, "{:.1f}x"),
    "bur_writeoff": ("written-off bureau lines", 3.0, "{:.0f}"),
}
IF_FEATURES = ["rep_avg_dpd", "rep_missed", "rep_max_dpd", "bur_avg_dpd", "bur_utilisation", "emi_to_income",
               "loan_to_income", "bur_writeoff_share", "balance_paydown"]


def _peer_groups(df: pd.DataFrame) -> pd.Series:
    tier = pd.cut(df["annual_income"], [0, 60_000, 100_000, 1e12], labels=["Lower income", "Middle income", "Upper income"], right=False)
    g = df["loan_type"].astype(str) + " / " + tier.astype(str)
    small = g.map(g.value_counts()) < MIN_PEER_N
    return g.where(~small, df["loan_type"].astype(str) + " / all incomes")


def detect(df: pd.DataFrame) -> pd.DataFrame:
    x = df.reset_index(drop=True).copy()
    n = len(x)
    x["peer_group"] = _peer_groups(x)
    peer_n = x["peer_group"].map(x["peer_group"].value_counts())
    score = np.zeros(n)
    reasons: list[list[str]] = [[] for _ in range(n)]
    types: list[set] = [set() for _ in range(n)]

    def add(mask: np.ndarray, weight: float, ftype: str, text_fn) -> None:
        nonlocal score
        idx = np.where(mask)[0]
        score[idx] += weight
        for i in idx:
            reasons[i].append(text_fn(i))
            types[i].add(ftype)

    # ---- A. own history ------------------------------------------------------------------------------
    rec, pri = x["rep_dpd_recent"].values, x["rep_dpd_prior"].values
    pstd = np.maximum(x["rep_dpd_prior_std"].fillna(0).values, 7.0)              # floor: a 'perfect' history is not infinitely tight
    z = (rec - pri) / pstd
    det = (z >= 3.0) & (rec >= 30)
    add(det, 3.0, "Own history", lambda i: f"Sudden deterioration: avg DPD over last 3 months is {rec[i]:.0f} days vs "
                                          f"{pri[i]:.1f} over its previous 9 months (z = {z[i]:.1f})")
    newd = (x["rep_delinquent_prior"].values == 0) & (x["rep_delinquent_recent"].values >= 2)
    add(newd, 3.0, "Own history", lambda i: f"New delinquency after a clean record: 0 late/missed months in months 1-9, "
                                           f"then {int(x['rep_delinquent_recent'].iloc[i])} of the last 3")
    streak = x["rep_longest_streak"].values >= 4
    add(streak, 2.5, "Own history", lambda i: f"Sustained delinquency: {int(x['rep_longest_streak'].iloc[i])} consecutive "
                                              "late/missed months")

    # ---- B. peer outliers ------------------------------------------------------------------------------
    for col, (label, floor, fmt) in PEER_METRICS.items():
        v = x[col]
        pct = v.groupby(x["peer_group"]).rank(pct=True, method="average")
        mask = ((pct >= PEER_PCTL) & (v >= floor)).fillna(False).values
        top = (100 * (1 - pct + 1 / peer_n)).clip(lower=0.1)                      # share of peers at or above this value
        add(mask, 1.5, "Peer outlier", lambda i, col=col, label=label, fmt=fmt, top=top:
            f"Peer outlier: {label} {fmt.format(x[col].iloc[i])} is in the top {top.iloc[i]:.1f}% of "
            f"{x['peer_group'].iloc[i]} peers (n={int(peer_n.iloc[i]):,})")

    # ---- C. bureau vs internal divergence ----------------------------------------------------------------
    stressed = ((x["bur_avg_dpd"] >= 30) | (x["bur_max_dpd"] >= 60)).fillna(False).values & (x["rep_delinquent"].values <= 1)
    add(stressed, 2.0, "Bureau divergence", lambda i: f"Stressed elsewhere, clean with us: bureau avg DPD {x['bur_avg_dpd'].iloc[i]:.0f} / max "
                                                    f"{x['bur_max_dpd'].iloc[i]:.0f} days, yet only {int(x['rep_delinquent'].iloc[i])} delinquent month(s) with us "
                                                    "(early-warning candidate)")
    clean_else = ((x["bur_lines"].values >= 1) & (x["bur_max_dpd"].fillna(99).values < 10) & (x["rep_missed"].values >= 3))
    add(clean_else, 2.0, "Bureau divergence", lambda i: f"Missing payments to us ({int(x['rep_missed'].iloc[i])} missed) while clean at other lenders "
                                                      f"(bureau max DPD {x['bur_max_dpd'].iloc[i]:.0f}): account-specific issue, dispute or strategic default?")

    # ---- D. multivariate isolation forest within peer group ------------------------------------------------
    iso = np.zeros(n)
    for g, idx in x.groupby("peer_group").groups.items():
        sub = x.loc[idx, IF_FEATURES].copy()
        sub["bur_avg_dpd"] = sub["bur_avg_dpd"].fillna(0)
        sub["bur_utilisation"] = sub["bur_utilisation"].fillna(sub["bur_utilisation"].median())
        sub = sub.fillna(sub.median())
        Z = StandardScaler().fit_transform(sub)
        forest = IsolationForest(n_estimators=200, contamination="auto", random_state=C.SEED, n_jobs=1).fit(Z)
        s = forest.decision_function(Z)                                           # lower = more anomalous
        iso[np.array(idx)] = pd.Series(s).rank(pct=True).values                   # percentile within segment
    mv = iso <= IF_TAIL
    add(mv, 2.0, "Multivariate outlier", lambda i: f"Unusual combination of repayment/bureau metrics vs {x['peer_group'].iloc[i]} peers "
                                                   f"(isolation-forest score in the most unusual {max(iso[i], 0.001):.1%})")

    x["anomaly_score"] = score
    x["flag_types"] = [", ".join(sorted(t)) for t in types]
    x["reasons"] = [" | ".join(r) for r in reasons]
    x["n_reasons"] = [len(r) for r in reasons]
    x["severity"] = np.select([score >= 5.5, score >= 4.0, score >= QUEUE_MIN_SCORE], ["High", "Medium", "Low"], default="")
    cols = ["account_id", "customer_id", "loan_type", "peer_group", "loan_amount", "anomaly_score", "severity", "flag_types",
            "n_reasons", "reasons", "rep_dpd_recent", "rep_dpd_prior", "rep_delinquent", "rep_missed", "rep_avg_dpd",
            "bur_avg_dpd", "bur_max_dpd", "bur_utilisation", "origination_year", "employment_type", "region"]
    q = x.loc[x["anomaly_score"] >= QUEUE_MIN_SCORE, cols + [C.TARGET]].copy()
    q["priority"] = q["anomaly_score"] * np.log10(q["loan_amount"])               # severity weighted by exposure
    return q.sort_values("priority", ascending=False).reset_index(drop=True)


def lift_check(q: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Post-hoc validation only: do flagged accounts default more often? (not an input to detection)"""
    flagged = df["account_id"].isin(q["account_id"])
    return {"flagged_accounts": int(flagged.sum()), "flag_rate": float(flagged.mean()),
            "default_rate_flagged": float(df.loc[flagged, C.TARGET].mean()),
            "default_rate_unflagged": float(df.loc[~flagged, C.TARGET].mean()),
            "lift": float(df.loc[flagged, C.TARGET].mean() / df.loc[~flagged, C.TARGET].mean())}


# --------------------------------------------------------------------------- human-in-the-loop store
DECISIONS_FILE = C.STATE_DIR / "review_decisions.csv"
DECISION_VALUES = ["Confirmed", "Dismissed", "Needs more info"]


def load_decisions(path=DECISIONS_FILE) -> pd.DataFrame:
    """Append-only log of every decision (full audit trail)."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp_utc", "account_id", "decision", "reviewer", "note"])


def save_decision(account_id: str, decision: str, reviewer: str = "analyst", note: str = "", path=DECISIONS_FILE) -> None:
    if decision not in DECISION_VALUES:
        raise ValueError(f"decision must be one of {DECISION_VALUES}")
    row = pd.DataFrame([{"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "account_id": account_id,
                         "decision": decision, "reviewer": reviewer or "analyst", "note": note}])
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def latest_status(decisions: pd.DataFrame) -> pd.Series:
    """account_id -> most recent decision."""
    if decisions.empty:
        return pd.Series(dtype=object)
    return decisions.sort_values("timestamp_utc").groupby("account_id")["decision"].last()
