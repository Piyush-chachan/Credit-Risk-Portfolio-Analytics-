import pandas as pd
from src import config as C
from src.ingestion import build_dataset, _norm


def _mini_raw(tmp_path, n_cust=60):
    raw = {t: pd.read_csv(C.RAW_DIR / f"{t}.csv") for t in C.FILE_CANDIDATES}
    cust = raw["customers"].head(n_cust)
    loans = raw["loan_accounts"][raw["loan_accounts"].customer_id.isin(cust.customer_id)]
    out = {"customers": cust, "loan_accounts": loans,
           "bureau_credit": raw["bureau_credit"][raw["bureau_credit"].customer_id.isin(cust.customer_id)],
           "repayment_history": raw["repayment_history"][raw["repayment_history"].account_id.isin(loans.account_id)],
           "collections_events": raw["collections_events"][raw["collections_events"].account_id.isin(loans.account_id)]}
    return out


def _write(tmp_path, tables):
    for k, v in tables.items():
        v.to_csv(tmp_path / f"{k}.csv", index=False)


def test_grain_is_one_row_per_account(tmp_path):
    t = _mini_raw(tmp_path)
    _write(tmp_path, t)
    b = build_dataset(tmp_path, save=False)
    assert len(b.dataset) == len(t["loan_accounts"])
    assert b.dataset.account_id.is_unique


def test_duplicates_orphans_and_gender_are_handled_and_logged(tmp_path):
    t = _mini_raw(tmp_path)
    t["customers"] = pd.concat([t["customers"], t["customers"].head(2)]).reset_index(drop=True)   # duplicate keys
    t["customers"].loc[[0, 1, 2], "gender"] = ["M", "F", "M"]
    orphan = t["bureau_credit"].head(1).assign(bureau_id="BUR_X", customer_id="CUST_NOPE")
    t["bureau_credit"] = pd.concat([t["bureau_credit"], orphan])
    _write(tmp_path, t)
    b = build_dataset(tmp_path, save=False)
    assert set(b.dataset.gender.unique()) <= {"Male", "Female"}
    dq = b.dq.set_index("check")
    assert dq.loc["Duplicate primary key ['customer_id']", "affected"] == 2
    assert dq.loc["Row whose customer_id has no parent", "affected"].max() >= 1
    assert len(b.dataset) == len(t["loan_accounts"])                                      # duplicates did not fan out


def test_missing_file_gives_clear_error(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path, save=False)


def test_real_data_register_finds_temporal_problems(built):
    from src.assets import load_json
    dq = {r["check"]: r for r in load_json("dq")}
    assert dq["Collections event dated BEFORE the loan was originated"]["affected"] > 1000
    assert dq["customer_since is AFTER the loan's origination_date"]["affected"] > 1000


def test_norm():
    assert _norm(" Days Past Due ") == "days_past_due"
