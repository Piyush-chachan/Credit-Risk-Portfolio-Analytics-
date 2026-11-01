"""Central configuration: paths, constants and artifact names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "models"
STATE_DIR = ROOT / "data" / "state"          # runtime state (review decisions, audit log) - git-ignored
for _d in (PROC_DIR, MODEL_DIR, STATE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "target_default"

# Accepted file names per table (tolerates the spaced names used in the brief's screenshots).
FILE_CANDIDATES = {
    "customers": ["customers.csv", "customer.csv"],
    "loan_accounts": ["loan_accounts.csv", "loan accounts.csv"],
    "bureau_credit": ["bureau_credit.csv", "bureau data.csv", "bureau.csv"],
    "repayment_history": ["repayment_history.csv", "repayments history.csv", "repayment history.csv"],
    "collections_events": ["collections_events.csv", "collection events.csv", "collections events.csv"],
}

# Business-rule thresholds used by data-quality checks.
MIN_PLAUSIBLE_AGE, MAX_PLAUSIBLE_AGE = 18, 90
MIN_WORKING_AGE = 16                  # years_employed cannot exceed age - 16
INCOME_FLOOR = 12_000                 # the data is censored at this value

# Risk segments are quantile bands of the out-of-fold predicted PD.
RISK_SEGMENT_CUTS = [0.40, 0.70, 0.90]            # Low <40%, Medium 40-70, High 70-90, Very High top 10%
RISK_SEGMENT_LABELS = ["Low", "Medium", "High", "Very High"]

# Artifact file names (all under data/processed).
ART = {
    "dataset": "analysis_ready.csv.gz",
    "collections": "collections_enriched.csv.gz",
    "scores": "account_scores.csv.gz",
    "drivers": "account_drivers.csv.gz",
    "anomalies": "anomaly_flags.csv.gz",
    "dq": "data_quality_register.json",
    "metrics": "model_metrics.json",
    "importance": "feature_importance.csv",
    "deciles": "score_deciles.csv",
    "curves": "model_curves.json",
    "psi": "psi_monitoring.csv",
    "ablation": "feature_ablation.csv",
    "insights": "portfolio_insights.json",
    "meta": "build_meta.json",
}
