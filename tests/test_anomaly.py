import pandas as pd
import pytest
from src import anomaly


def test_label_is_never_used(portfolio):
    q1 = anomaly.detect(portfolio)
    shuffled = portfolio.copy()
    shuffled["target_default"] = shuffled["target_default"].sample(frac=1, random_state=1).values
    q2 = anomaly.detect(shuffled)
    assert set(q1.account_id) == set(q2.account_id)


def test_every_flag_has_reasons_and_severity(portfolio):
    q = anomaly.detect(portfolio)
    assert len(q) > 0 and (q.reasons.str.len() > 10).all() and q.severity.isin(["Low", "Medium", "High"]).all()
    assert 0.01 < len(q) / len(portfolio) < 0.15                        # a reviewable queue, not everything


def test_decision_store_is_append_only(tmp_path):
    p = tmp_path / "d.csv"
    anomaly.save_decision("ACC1", "Confirmed", "a", "note", path=p)
    anomaly.save_decision("ACC1", "Dismissed", "b", "changed mind", path=p)
    d = anomaly.load_decisions(p)
    assert len(d) == 2 and anomaly.latest_status(d)["ACC1"] == "Dismissed"
    with pytest.raises(ValueError):
        anomaly.save_decision("ACC1", "Maybe", path=p)
