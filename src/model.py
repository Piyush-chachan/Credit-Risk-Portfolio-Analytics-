"""Stage 3 - default model.

Design choices (each one answers a specific weakness found in a naive build):

* **Customer-grouped cross-validation.** 14,685 loans belong to 10,000 customers and bureau features are
  shared by a customer's loans, so a row-level split leaks the same customer into train and test.
  We use StratifiedGroupKFold on customer_id.
* **Out-of-fold (OOF) scores.** Every account's displayed PD comes from a model that never saw it.
* **No class re-weighting** -> outputs are genuine probabilities of default (PDs), not "balanced" scores.
* **Collections excluded** from features (recorded after delinquency; dates unreliable). An ablation proves
  excluding them costs nothing.
* **Protected attributes** (gender, marital status) are not used as predictors.
* **Per-account drivers that reconcile with the score.** For logistic regression the logit decomposes
  exactly: logit = intercept + sum(coef_j * z_j). Each driver = coef_j * (z_j - mean_j), grouped back to the
  original feature, so drivers sum to (logit - average logit).
* Logistic regression is the *champion* unless the ML challenger beats it by > 0.005 AUC (interpretability
  is worth a small AUC gap in credit risk).
"""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- feature contract
ORIGINATION = ["age", "log_income", "years_employed", "num_dependents", "log_loan_amount", "interest_rate_pct",
               "tenure_months", "collateral_flag", "log_loan_to_income", "years_employed_missing",
               "num_dependents_missing", "interest_rate_missing"]
BUREAU = ["bur_lines", "bur_active", "bur_writeoff", "bur_writeoff_share", "bur_max_dpd", "bur_avg_dpd",
          "bur_utilisation", "log_bur_outstanding_to_income", "bur_thin_file", "bur_nbfc_fintech_share"]
REPAYMENT_FULL = ["rep_late", "rep_missed", "rep_delinquent", "rep_avg_dpd", "rep_max_dpd", "rep_longest_streak",
                  "rep_delinquent_early", "rep_delinquent_recent", "log_emi_to_income", "balance_paydown"]
REPAYMENT_EARLY = ["rep_delinquent_early", "rep_missed_early", "rep_avg_dpd_early"]     # months 1-6 only
CATEGORICAL = ["loan_type", "employment_type", "education_level"]
COLLECTIONS = ["col_events", "col_recovered"]                                              # excluded; ablation only

FEATURE_SETS = {
    "Origination only (customer + loan)": ORIGINATION,
    "+ Bureau": ORIGINATION + BUREAU,
    "+ Bureau + Repayment months 1-12  (FINAL)": ORIGINATION + BUREAU + REPAYMENT_FULL,
    "Early-window: Repayment months 1-6 only": ORIGINATION + BUREAU + REPAYMENT_EARLY,
    "FINAL + Collections (excluded: post-delinquency)": ORIGINATION + BUREAU + REPAYMENT_FULL + COLLECTIONS,
}
FINAL_KEY = "+ Bureau + Repayment months 1-12  (FINAL)"
NUMERIC = ORIGINATION + BUREAU + REPAYMENT_FULL

LABELS = {
    "age": "Age", "log_income": "Annual income", "years_employed": "Years employed", "num_dependents": "Dependents",
    "log_loan_amount": "Loan amount", "interest_rate_pct": "Interest rate", "tenure_months": "Tenure",
    "collateral_flag": "Secured loan", "log_loan_to_income": "Loan-to-income", "years_employed_missing": "Years employed missing",
    "num_dependents_missing": "Dependents missing", "interest_rate_missing": "Rate missing",
    "bur_lines": "Bureau lines", "bur_active": "Active bureau lines", "bur_writeoff": "Written-off bureau lines",
    "bur_writeoff_share": "Written-off share of bureau lines", "bur_max_dpd": "Bureau max DPD", "bur_avg_dpd": "Bureau avg DPD",
    "bur_utilisation": "Bureau utilisation", "log_bur_outstanding_to_income": "Bureau debt-to-income",
    "bur_thin_file": "Thin bureau file", "bur_nbfc_fintech_share": "NBFC/Fintech share of lines",
    "rep_late": "Late months", "rep_missed": "Missed months", "rep_delinquent": "Delinquent months (late+missed)",
    "rep_avg_dpd": "Avg DPD (12m)", "rep_max_dpd": "Max DPD (12m)", "rep_longest_streak": "Longest delinquent streak",
    "rep_delinquent_early": "Delinquent months 1-6", "rep_delinquent_recent": "Delinquent months 10-12",
    "log_emi_to_income": "EMI-to-income", "balance_paydown": "Balance paid down",
    "rep_missed_early": "Missed months 1-6", "rep_avg_dpd_early": "Avg DPD months 1-6",
    "col_events": "Collections events", "col_recovered": "Collections recovered",
    "loan_type": "Loan type", "employment_type": "Employment type", "education_level": "Education",
}


def design(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic transforms (no fitting -> no leakage): logs for skewed money/ratio fields."""
    x = pd.DataFrame(index=df.index)
    raw_cols = set(ORIGINATION + BUREAU + REPAYMENT_FULL + REPAYMENT_EARLY + COLLECTIONS) - {
        "log_income", "log_loan_amount", "log_loan_to_income", "log_bur_outstanding_to_income", "log_emi_to_income"}
    for c in raw_cols:
        x[c] = df[c].astype("float64")
    x["log_income"] = np.log(df["annual_income"].clip(lower=1))
    x["log_loan_amount"] = np.log(df["loan_amount"].clip(lower=1))
    x["log_loan_to_income"] = np.log1p(df["loan_to_income"].clip(lower=0))
    x["log_bur_outstanding_to_income"] = np.log1p(df["bur_outstanding_to_income"].clip(lower=0))
    x["log_emi_to_income"] = np.log1p(df["emi_to_income"].clip(lower=0))
    for c in CATEGORICAL:
        x[c] = df[c].astype(str)
    return x


def _preprocessor(num: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def _lr():
    return LogisticRegressionCV(Cs=[0.003, 0.01, 0.03, 0.1], cv=3, scoring="roc_auc", max_iter=3000, random_state=C.SEED)


def _gbm():
    import lightgbm as lgb    # lazy: the dashboard never imports it when pre-built artifacts exist
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=5, min_child_samples=80,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0,
                              random_state=C.SEED, verbose=-1, n_jobs=1)


# --------------------------------------------------------------------------- metrics
def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p, n_boot: int = 300, seed: int = C.SEED) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"AUC": float(auc), "AUC_CI95": [float(lo), float(hi)], "KS": ks_stat(y, p), "Gini": float(2 * auc - 1)}


# --------------------------------------------------------------------------- grouping helpers
def _source_feature(col_name: str) -> str:
    """Map a transformed column back to its original feature (so one-hot dummies are grouped)."""
    if col_name.startswith("num__"):
        return col_name[5:]
    body = col_name[5:]
    for c in CATEGORICAL:
        if body.startswith(c + "_"):
            return c
    return body


def _group_matrix(M: np.ndarray, names: list[str], groups: list[str]) -> pd.DataFrame:
    src = np.array([_source_feature(n) for n in names])
    return pd.DataFrame({g: M[:, src == g].sum(axis=1) for g in groups})


# --------------------------------------------------------------------------- cross-validation
def cross_validate(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, num: list[str], with_gbm: bool = True,
                   with_contrib: bool = True) -> dict:
    """Customer-grouped CV. Returns OOF predictions, per-account driver matrices, GBM SHAP importance."""
    cv = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    oof_lr, oof_gb = np.zeros(len(X)), np.zeros(len(X))
    glist = num + CATEGORICAL
    lr_contrib = pd.DataFrame(0.0, index=range(len(X)), columns=glist)
    gb_contrib = pd.DataFrame(0.0, index=range(len(X)), columns=glist)
    shap_abs = pd.Series(0.0, index=glist)
    fold_auc = []
    for tr, te in cv.split(X, y, groups):
        pre = _preprocessor(num).fit(X.iloc[tr])
        Ztr, Zte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
        names = list(pre.get_feature_names_out())
        lr = _lr().fit(Ztr, y[tr])
        oof_lr[te] = lr.predict_proba(Zte)[:, 1]
        fold_auc.append(roc_auc_score(y[te], oof_lr[te]))
        if with_contrib:
            lr_contrib.iloc[te] = _group_matrix((Zte - Ztr.mean(axis=0)) * lr.coef_[0], names, glist).values
        if with_gbm:
            gb = _gbm().fit(Ztr, y[tr])
            oof_gb[te] = gb.predict_proba(Zte)[:, 1]
            if with_contrib:
                contrib = gb.booster_.predict(Zte, pred_contrib=True)[:, :-1]      # exact TreeSHAP (log-odds), last col = bias
                g = _group_matrix(contrib, names, glist)
                gb_contrib.iloc[te] = g.values
                shap_abs += g.abs().sum()
    out = {"oof_lr": oof_lr, "oof_gb": oof_gb if with_gbm else None, "fold_auc_lr": fold_auc}
    if with_contrib:
        out.update(lr_contrib=lr_contrib, gb_contrib=gb_contrib, shap_importance=(shap_abs / len(X)).sort_values(ascending=False))
    return out


def out_of_time(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, num: list[str], cut_year: int = 2021) -> dict:
    """Train on loans originated <= cut_year, test on later originations."""
    yr = df["origination_date"].dt.year.values
    tr, te = np.where(yr <= cut_year)[0], np.where(yr > cut_year)[0]
    pre = _preprocessor(num).fit(X.iloc[tr])
    Ztr, Zte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    res = {"train_rows": int(len(tr)), "test_rows": int(len(te)), "split": f"train <= {cut_year}, test > {cut_year}"}
    res["Logistic Regression"] = metrics(y[te], _lr().fit(Ztr, y[tr]).predict_proba(Zte)[:, 1], n_boot=100)
    res["LightGBM"] = metrics(y[te], _gbm().fit(Ztr, y[tr]).predict_proba(Zte)[:, 1], n_boot=100)
    return res


# --------------------------------------------------------------------------- monitoring: PSI
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with reference-quantile bins. <0.10 stable, 0.10-0.25 watch, >0.25 act."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_table(df: pd.DataFrame, X: pd.DataFrame, score: np.ndarray, features: list[str], cut_year: int = 2021) -> pd.DataFrame:
    yr = df["origination_date"].dt.year.values
    ref, cur = yr <= cut_year, yr > cut_year
    rows = [{"item": "Model score (PD)", "type": "score", "PSI": psi(score[ref], score[cur])}]
    for f in features:
        rows.append({"item": LABELS.get(f, f), "type": "feature", "PSI": psi(X.loc[ref, f], X.loc[cur, f])})
    t = pd.DataFrame(rows)
    t["status"] = pd.cut(t["PSI"], [-1, 0.10, 0.25, 99], labels=["Stable", "Watch", "Action"]).astype(str)
    return t.sort_values("PSI", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- segments + deciles
def segment_thresholds(pd_oof: np.ndarray) -> list[float]:
    return [float(np.quantile(pd_oof, q)) for q in C.RISK_SEGMENT_CUTS]


def assign_segment(p: np.ndarray, thresholds: list[float]) -> np.ndarray:
    return np.array(C.RISK_SEGMENT_LABELS)[np.searchsorted(thresholds, p, side="right")]


def decile_table(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)   # 1 = riskiest
    t = d.groupby("decile").agg(accounts=("y", "size"), defaults=("y", "sum"), avg_pd=("p", "mean"), actual_rate=("y", "mean")).reset_index()
    t["lift"] = t["actual_rate"] / d["y"].mean()
    t["cum_bad_capture"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["cum_good_capture"] = (t["accounts"] - t["defaults"]).cumsum() / (t["accounts"] - t["defaults"]).sum()
    t["KS_at_decile"] = t["cum_bad_capture"] - t["cum_good_capture"]
    return t


# --------------------------------------------------------------------------- training entry point
@dataclass
class TrainResult:
    scores: pd.DataFrame
    drivers: pd.DataFrame
    metrics: dict
    importance: pd.DataFrame
    deciles: pd.DataFrame
    curves: dict
    psi: pd.DataFrame
    ablation: pd.DataFrame


def train(df: pd.DataFrame, save: bool = True) -> TrainResult:
    d = df[df[C.TARGET].isin([0, 1])].reset_index(drop=True)
    X, y, groups = design(d), d[C.TARGET].astype(int).values, d["customer_id"].values
    num = FEATURE_SETS[FINAL_KEY]

    # --- ablation (logistic regression only: fast, comparable) -----------------------------------
    abl = []
    for name, feats in FEATURE_SETS.items():
        cvr = cross_validate(X, y, groups, feats, with_gbm=False, with_contrib=False)
        abl.append({"feature_set": name, "n_numeric_features": len(feats), "AUC_LR": roc_auc_score(y, cvr["oof_lr"]),
                    "KS_LR": ks_stat(y, cvr["oof_lr"])})
    ablation = pd.DataFrame(abl)

    # --- final CV with both models ---------------------------------------------------------------
    cv = cross_validate(X, y, groups, num)
    m_lr, m_gb = metrics(y, cv["oof_lr"]), metrics(y, cv["oof_gb"])
    champion = "LightGBM" if m_gb["AUC"] - m_lr["AUC"] > 0.005 else "Logistic Regression"
    p_champ = cv["oof_gb"] if champion == "LightGBM" else cv["oof_lr"]
    drivers = cv["gb_contrib"] if champion == "LightGBM" else cv["lr_contrib"]

    oot = out_of_time(d, X, y, num)
    thresholds = segment_thresholds(p_champ)
    seg = assign_segment(p_champ, thresholds)
    deciles = decile_table(y, p_champ)
    psi_t = psi_table(d, X, p_champ, ["log_income", "age", "rep_delinquent", "rep_avg_dpd", "bur_avg_dpd", "log_loan_amount",
                                      "log_loan_to_income", "bur_writeoff", "interest_rate_pct", "log_emi_to_income"])

    # --- early-window robustness (leakage check) --------------------------------------------------
    early = ablation.set_index("feature_set").loc["Early-window: Repayment months 1-6 only", "AUC_LR"]

    # --- importance: LR standardised |coef| (final model) + GBM SHAP -----------------------------
    pre = _preprocessor(num).fit(X)
    Z = pre.transform(X)
    names = list(pre.get_feature_names_out())
    lr_final = _lr().fit(Z, y)
    gb_final = _gbm().fit(Z, y)
    coef = pd.DataFrame({"feature": [_source_feature(n) for n in names], "coef": lr_final.coef_[0]})
    lr_imp = coef.assign(abs_coef=coef["coef"].abs()).groupby("feature")["abs_coef"].sum()
    # direction: for numeric = sign of coefficient; for categorical = n/a
    sign = coef[coef["feature"].isin(num)].set_index("feature")["coef"]
    importance = pd.DataFrame({"feature": cv["shap_importance"].index, "mean_abs_shap_gbm": cv["shap_importance"].values})
    importance["abs_coef_lr"] = importance["feature"].map(lr_imp)
    importance["mean_abs_driver"] = importance["feature"].map(drivers.abs().mean())     # champion's avg |log-odds impact|
    importance["lr_direction"] = importance["feature"].map(lambda f: "higher -> riskier" if sign.get(f, 0) > 0 else
                                                           ("higher -> safer" if f in sign.index else "categorical"))
    importance["label"] = importance["feature"].map(lambda f: LABELS.get(f, f))
    importance = importance.sort_values("mean_abs_driver", ascending=False).reset_index(drop=True)

    fpr, tpr, _ = roc_curve(y, p_champ)
    fpr_g, tpr_g, _ = roc_curve(y, cv["oof_gb"])
    step = max(1, len(fpr) // 200)
    curves = {"roc_lr": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
              "roc_gb": {"fpr": fpr_g[::max(1, len(fpr_g) // 200)].tolist(), "tpr": tpr_g[::max(1, len(fpr_g) // 200)].tolist()}}

    prev = float(y.mean())
    met = {
        "champion": champion, "n_accounts": int(len(y)), "n_customers": int(pd.Series(groups).nunique()),
        "default_rate": prev, "cv": f"{C.N_FOLDS}-fold StratifiedGroupKFold by customer_id (pooled out-of-fold)",
        "Logistic Regression": {**m_lr, "fold_auc_mean": float(np.mean(cv["fold_auc_lr"])), "fold_auc_std": float(np.std(cv["fold_auc_lr"]))},
        "LightGBM": m_gb, "out_of_time": oot, "early_window_auc_lr": float(early),
        "final_features_numeric": len(num), "final_features_categorical": len(CATEGORICAL),
        "segment_thresholds": thresholds,
        "selection_rule": "ML challenger promoted only if AUC gain > 0.005 over the logistic baseline",
    }
    scores = pd.DataFrame({"account_id": d["account_id"], "pd_score": p_champ, "risk_segment": seg,
                           "pd_lr": cv["oof_lr"], "pd_gbm": cv["oof_gb"]})
    scores["risk_segment"] = pd.Categorical(scores["risk_segment"], C.RISK_SEGMENT_LABELS, ordered=True)
    drivers = drivers.assign(account_id=d["account_id"].values)
    drivers = drivers[["account_id"] + [c for c in drivers.columns if c != "account_id"]]

    if save:
        joblib.dump({"pre": pre, "lr": lr_final, "gbm": gb_final, "numeric": num, "categorical": CATEGORICAL,
                     "champion": champion, "thresholds": thresholds, "train_means": Z.mean(axis=0), "names": names,
                     "version": "1.0"}, C.MODEL_DIR / "model.joblib")
        seg_out = scores.copy()
        seg_out["risk_segment"] = seg_out["risk_segment"].astype(str)
        seg_out.round(6).to_csv(C.PROC_DIR / C.ART["scores"], index=False)
        drivers.round(5).to_csv(C.PROC_DIR / C.ART["drivers"], index=False)
        importance.to_csv(C.PROC_DIR / C.ART["importance"], index=False)
        deciles.to_csv(C.PROC_DIR / C.ART["deciles"], index=False)
        psi_t.to_csv(C.PROC_DIR / C.ART["psi"], index=False)
        ablation.to_csv(C.PROC_DIR / C.ART["ablation"], index=False)
        (C.PROC_DIR / C.ART["metrics"]).write_text(json.dumps(met, indent=1))
        (C.PROC_DIR / C.ART["curves"]).write_text(json.dumps(curves))
    return TrainResult(scores, drivers, met, importance, deciles, curves, psi_t, ablation)


# --------------------------------------------------------------------------- scoring new accounts
def score_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """Score accounts with the saved final model (what a production batch job would run)."""
    art = joblib.load(C.MODEL_DIR / "model.joblib")
    X = design(df)
    Z = art["pre"].transform(X[art["numeric"] + art["categorical"]])
    model = art["gbm"] if art["champion"] == "LightGBM" else art["lr"]
    p = model.predict_proba(Z)[:, 1]
    out = pd.DataFrame({"account_id": df["account_id"].values, "pd_score": p,
                        "risk_segment": assign_segment(p, art["thresholds"])})
    return out


def top_drivers(contrib_row: pd.Series, n_up: int = 3, n_down: int = 2) -> tuple[list, list]:
    c = contrib_row.drop(labels=["account_id"], errors="ignore").sort_values()
    return list(c[c > 0].sort_values(ascending=False).head(n_up).items()), list(c[c < 0].head(n_down).items())


def describe_driver(feature: str, row: pd.Series) -> str:
    """Human-readable 'what the value is' for a driver, e.g. 'Annual income = $34,000'."""
    label = LABELS.get(feature, feature)
    try:
        if feature in CATEGORICAL:
            return f"{label} = {row[feature]}"
        m = {"log_income": ("annual_income", "${:,.0f}"), "log_loan_amount": ("loan_amount", "${:,.0f}"),
             "log_loan_to_income": ("loan_to_income", "{:.2f}x"), "log_emi_to_income": ("emi_to_income", "{:.1%}"),
             "log_bur_outstanding_to_income": ("bur_outstanding_to_income", "{:.2f}x"),
             "bur_utilisation": ("bur_utilisation", "{:.0%}"), "balance_paydown": ("balance_paydown", "{:.0%}"),
             "interest_rate_pct": ("interest_rate_pct", "{:.1f}%"), "bur_writeoff_share": ("bur_writeoff_share", "{:.0%}"),
             "bur_nbfc_fintech_share": ("bur_nbfc_fintech_share", "{:.0%}")}
        col, fmt = m.get(feature, (feature, "{:,.1f}"))
        v = row[col]
        return f"{label} = {fmt.format(v)}" if pd.notna(v) else f"{label} = missing"
    except Exception:
        return label
