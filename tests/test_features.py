import pandas as pd
from src.features import repayment_features, bureau_features


def _rep(statuses, acc="A1"):
    return pd.DataFrame({"account_id": acc, "payment_month": range(1, len(statuses) + 1),
                         "payment_status": statuses, "emi_due": 100.0,
                         "emi_paid": [0.0 if s == "Missed" else 100.0 for s in statuses],
                         "days_past_due": [0 if s == "On-Time" else (45 if s == "Missed" else 10) for s in statuses],
                         "outstanding_balance": [1000 - 10 * i for i in range(len(statuses))]})


def test_longest_streak_and_windows():
    s = ["On-Time", "Late", "Late", "Missed", "On-Time", "On-Time", "Late", "On-Time", "On-Time", "Late", "Late", "Late"]
    f = repayment_features(_rep(s)).iloc[0]
    assert f.rep_longest_streak == 3 and f.rep_delinquent == 7 and f.rep_missed == 1
    assert f.rep_delinquent_recent == 3 and f.rep_delinquent_prior == 4 and f.rep_delinquent_early == 3


def test_clean_account_has_zero_streak():
    f = repayment_features(_rep(["On-Time"] * 12)).iloc[0]
    assert f.rep_longest_streak == 0 and f.rep_delinquent == 0


def test_bureau_aggregation_one_row_per_customer():
    b = pd.DataFrame({"bureau_id": ["1", "2", "3"], "customer_id": ["C1", "C1", "C2"],
                      "account_status": ["Active", "Written-Off", "Closed"], "lender_category": ["Bank", "NBFC", "Fintech"],
                      "sanctioned_amount": [100.0, 200.0, 50.0], "outstanding_amount": [50.0, 100.0, 0.0],
                      "days_past_due": [10.0, None, 0.0]})
    out = bureau_features(b).set_index("customer_id")
    assert out.loc["C1", "bur_lines"] == 2 and out.loc["C1", "bur_writeoff"] == 1
    assert out.loc["C1", "bur_max_dpd"] == 10                                   # NaN skipped, not treated as 0
    assert out.loc["C1", "bur_utilisation"] == 0.5
