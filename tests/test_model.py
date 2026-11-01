import json
import numpy as np
import pandas as pd
from src import config as C, model
from src.assets import load_json, load_csv


def test_psi_identical_is_zero_and_shift_is_detected():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert model.psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert model.psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_ks_and_segments():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert model.ks_stat(y, np.array([.1, .2, .3, .7, .8, .9])) == 1.0
    th = [0.1, 0.2, 0.3]
    assert list(model.assign_segment(np.array([.05, .15, .25, .5]), th)) == C.RISK_SEGMENT_LABELS


def test_feature_contract_has_no_leaky_or_protected_fields():
    feats = set(model.NUMERIC) | set(model.CATEGORICAL)
    assert not any(f.startswith("col_") for f in feats)                 # collections excluded
    assert not {"gender", "marital_status", "target_default", "customer_since"} & feats


def test_reported_metrics_are_plausible_and_consistent(built):
    m = load_json("metrics")
    lr = m["Logistic Regression"]
    assert 0.60 < lr["AUC"] < 0.80 and abs(lr["Gini"] - (2 * lr["AUC"] - 1)) < 1e-9
    assert 0 < lr["KS"] < 1 and lr["AUC_CI95"][0] < lr["AUC"] < lr["AUC_CI95"][1]


def test_segments_are_calibrated_and_monotonic(portfolio):
    t = portfolio.groupby("risk_segment").agg(pd=("pd_score", "mean"), act=(C.TARGET, "mean")).loc[C.RISK_SEGMENT_LABELS]
    assert t["act"].is_monotonic_increasing
    assert (abs(t["pd"] - t["act"]) < 0.03).all()                       # predicted PD ~ realised rate


def test_drivers_reconcile_with_scores(built):
    d = load_csv("drivers").set_index("account_id")
    s = load_csv("scores").set_index("account_id").loc[d.index]
    logit = np.log(s.pd_lr / (1 - s.pd_lr))
    assert np.corrcoef(d.sum(axis=1), logit - logit.mean())[0, 1] > 0.99


def test_saved_model_scores_new_accounts(portfolio):
    out = model.score_accounts(portfolio.head(200))
    assert out.pd_score.between(0, 1).all() and set(out.risk_segment) <= set(C.RISK_SEGMENT_LABELS)
