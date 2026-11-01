"""Build and load every pipeline artifact (ingestion -> analytics -> model -> anomalies -> insights)."""
from __future__ import annotations
import json
import time

import pandas as pd

from . import config as C
from . import analytics, anomaly, model
from .ingestion import build_dataset


def build_all(save: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    log("[1/5] Ingestion: load, data-quality register, join ...")
    b = build_dataset(save=save)
    df = b.dataset
    log(f"      {len(df):,} account rows | {int((b.dq.status == 'Issue').sum())} data-quality issues logged of {len(b.dq)} checks")
    log("[2/5] Model: grouped CV, LR + LightGBM, ablation, out-of-time, PSI ...")
    res = model.train(df, save=save)
    m = res.metrics
    log(f"      champion={m['champion']} | AUC={m['Logistic Regression']['AUC']:.3f} KS={m['Logistic Regression']['KS']:.3f} "
        f"Gini={m['Logistic Regression']['Gini']:.3f} | LightGBM AUC={m['LightGBM']['AUC']:.3f}")
    full = df.merge(res.scores[["account_id", "pd_score", "risk_segment"]], on="account_id", how="left")
    full["risk_segment"] = full["risk_segment"].astype(str)
    log("[3/5] Anomaly detection ...")
    q = anomaly.detect(full)
    lift = anomaly.lift_check(q, full)
    log(f"      {len(q):,} accounts in review queue ({lift['flag_rate']:.1%}); default-rate lift vs unflagged {lift['lift']:.2f}x")
    log("[4/5] Portfolio insights ...")
    ins = analytics.compute_insights(full)
    meta = {"built_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"), "accounts": int(len(df)),
            "anomaly_lift": lift, "seconds": round(time.time() - t0, 1)}
    if save:
        q.to_csv(C.PROC_DIR / C.ART["anomalies"], index=False)
        (C.PROC_DIR / C.ART["insights"]).write_text(json.dumps(ins, indent=1))
        (C.PROC_DIR / C.ART["meta"]).write_text(json.dumps(meta, indent=1))
    log(f"[5/5] Done in {meta['seconds']}s. Artifacts in {C.PROC_DIR}")
    return {"dataset": full, "dq": b.dq, "metrics": m, "anomalies": q, "insights": ins}


def artifacts_ready() -> bool:
    return all((C.PROC_DIR / C.ART[k]).exists() for k in C.ART) and (C.MODEL_DIR / "model.joblib").exists()


def ensure_built() -> None:
    if not artifacts_ready():
        build_all(save=True, verbose=False)


def load_json(key: str):
    return json.loads((C.PROC_DIR / C.ART[key]).read_text())


def load_csv(key: str) -> pd.DataFrame:
    return pd.read_csv(C.PROC_DIR / C.ART[key])


def load_portfolio() -> pd.DataFrame:
    """Analysis-ready dataset + model score + risk segment (one row per account)."""
    ds = pd.read_csv(C.PROC_DIR / C.ART["dataset"], parse_dates=["origination_date", "customer_since"])
    sc = load_csv("scores")[["account_id", "pd_score", "risk_segment"]]
    df = ds.merge(sc, on="account_id", how="left")
    for c in ("origination_year",):
        df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True)
    return df
