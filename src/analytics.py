"""Stage 2 - portfolio analytics: segment profiles, cross-dimensional cuts, vintage trend, insights.

Definitions (used identically by the dashboard, the Ask-the-Portfolio assistant and the PDFs)
  default rate            = defaulted accounts / accounts
  exposure                = sum of ORIGINAL loan_amount (no current-outstanding field exists on the loan table)
  exposure default rate   = loan_amount of defaulted accounts / total loan_amount (value-weighted)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

DIMENSIONS = {          # column -> display label (analyst-facing dimensions)
    "loan_type": "Loan type", "employment_type": "Employment type", "income_band": "Income band", "age_band": "Age band",
    "region": "Region", "city_tier": "City tier", "education_level": "Education", "housing_type": "Housing",
    "disbursal_channel": "Disbursal channel", "loan_purpose": "Loan purpose", "occupation_sector": "Occupation sector",
    "secured": "Secured / unsecured", "bureau_profile": "Bureau profile", "risk_segment": "Model risk segment",
    "origination_year": "Origination year", "origination_quarter": "Origination quarter", "gender": "Gender", "marital_status": "Marital status",
}
ORDERS = {
    "income_band": ["<40k", "40-60k", "60-80k", "80-100k", "100k+"],
    "age_band": ["<30", "30-39", "40-49", "50-59", "60+"],
    "risk_segment": C.RISK_SEGMENT_LABELS,
}


def wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


def _order(dim: str, values) -> list:
    return [v for v in ORDERS[dim] if v in set(values)] if dim in ORDERS else sorted(values, key=str)


def segment_table(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    d = df.assign(_d_exp=df["loan_amount"] * df[C.TARGET])
    agg = {"accounts": ("account_id", "size"), "defaults": (C.TARGET, "sum"), "exposure": ("loan_amount", "sum"),
           "defaulted_exposure": ("_d_exp", "sum")}
    if "pd_score" in d:
        agg["avg_pd"] = ("pd_score", "mean")
    t = d.groupby(dim, observed=True).agg(**agg).reset_index()
    t["default_rate"] = t["defaults"] / t["accounts"]
    t["exposure_default_rate"] = t["defaulted_exposure"] / t["exposure"]
    t["account_share"] = t["accounts"] / t["accounts"].sum()
    t["exposure_share"] = t["exposure"] / t["exposure"].sum()
    t["ci_low"], t["ci_high"] = wilson(t["defaults"], t["accounts"])
    t[dim] = pd.Categorical(t[dim], _order(dim, t[dim]), ordered=True)
    return t.sort_values(dim).reset_index(drop=True)


def cross_table(df: pd.DataFrame, rows: str, cols: str, min_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Default-rate matrix + count matrix. Cells under min_n are blanked (too small to read)."""
    rate = df.pivot_table(index=rows, columns=cols, values=C.TARGET, aggfunc="mean", observed=True)
    n = df.pivot_table(index=rows, columns=cols, values="account_id", aggfunc="count", observed=True)
    rate = rate.where(n >= min_n)
    ri, ci = _order(rows, rate.index), _order(cols, rate.columns)
    return rate.loc[ri, ci], n.loc[ri, ci]


def vintage_trend(df: pd.DataFrame, by: str = "origination_quarter") -> pd.DataFrame:
    t = df.groupby(by).agg(accounts=("account_id", "size"), defaults=(C.TARGET, "sum"), exposure=("loan_amount", "sum")).reset_index()
    t["default_rate"] = t["defaults"] / t["accounts"]
    return t.sort_values(by).reset_index(drop=True)


def kpis(df: pd.DataFrame) -> dict:
    dexp = float((df["loan_amount"] * df[C.TARGET]).sum())
    exp = float(df["loan_amount"].sum())
    return {"accounts": int(len(df)), "customers": int(df["customer_id"].nunique()), "exposure": exp,
            "default_rate": float(df[C.TARGET].mean()) if len(df) else float("nan"),
            "exposure_default_rate": dexp / exp if exp else float("nan"), "defaulted_exposure": dexp,
            "avg_pd": float(df["pd_score"].mean()) if "pd_score" in df and len(df) else float("nan")}


# --------------------------------------------------------------------------- insights
def _pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _money(x: float) -> str:
    return f"${x / 1e6:,.1f}M" if x >= 1e6 else f"${x:,.0f}"


def compute_insights(df: pd.DataFrame) -> list[dict]:
    """Five insights, each with evidence (exact figures + n) and a 'so what'. Numbers are computed, never typed."""
    out = []
    overall = df[C.TARGET].mean()
    k = kpis(df)

    # 1. Mortgage concentration -------------------------------------------------------------------------
    lt = segment_table(df, "loan_type").set_index("loan_type")
    m = lt.loc["Mortgage"]
    share_def_exp = m["defaulted_exposure"] / lt["defaulted_exposure"].sum()
    out.append({
        "id": "mortgage_concentration", "cross_dimensional": False, "dimension": "loan type x exposure",
        "headline": f"Mortgages are {_pct(m['account_share'])} of accounts but {_pct(m['exposure_share'], 0)} of exposure, and default most",
        "evidence": (f"Mortgage default rate {_pct(m['default_rate'])} (n={int(m['accounts']):,}) vs {_pct(overall)} portfolio; "
                     f"{_pct(share_def_exp, 0)} of all defaulted exposure ({_money(m['defaulted_exposure'])} of {_money(k['defaulted_exposure'])}). "
                     f"Value-weighted default rate is {_pct(k['exposure_default_rate'])} against {_pct(overall)} by count."),
        "so_what": "Headline default rate understates dollar risk. Mortgage underwriting, LTV limits and early-arrears monitoring are the "
                   "highest-leverage controls; report the exposure-weighted rate alongside the count-based one.",
        "numbers": {"mortgage_default_rate": float(m["default_rate"]), "mortgage_exposure_share": float(m["exposure_share"]),
                    "mortgage_account_share": float(m["account_share"]), "share_defaulted_exposure": float(share_def_exp)}})

    # 2. Employment x income (cross-dimensional) -------------------------------------------------------
    et = segment_table(df, "employment_type").set_index("employment_type")
    un = et.loc["Unemployed"]
    rate, n = cross_table(df, "employment_type", "income_band", min_n=100)
    lo = df[(df["income_band"] == "<40k")]
    un_lo = lo[lo["employment_type"] == "Unemployed"][C.TARGET]
    sal_lo = lo[lo["employment_type"] == "Salaried"][C.TARGET]
    mid = df[(df["income_band"] == "40-60k")]
    un_mid, sal_mid = (mid[mid["employment_type"] == t][C.TARGET] for t in ("Unemployed", "Salaried"))
    inc = segment_table(df, "income_band").set_index("income_band")
    out.append({
        "id": "employment_income", "cross_dimensional": True, "dimension": "employment type x income band",
        "headline": f"Unemployed borrowers default at {_pct(un['default_rate'], 0)}, and low income compounds it",
        "evidence": (f"Unemployed: {_pct(un['default_rate'])} (n={int(un['accounts']):,}, {_pct(un['account_share'])} of accounts). "
                     f"Income <40k: unemployed {_pct(un_lo.mean())} (n={len(un_lo)}) vs salaried {_pct(sal_lo.mean())} (n={len(sal_lo)}); "
                     f"40-60k: {_pct(un_mid.mean())} (n={len(un_mid)}) vs {_pct(sal_mid.mean())} (n={len(sal_mid)}). "
                     f"Across all employment, <40k income defaults at {_pct(inc.loc['<40k', 'default_rate'])} vs {_pct(inc.loc['100k+', 'default_rate'])} at 100k+."),
        "so_what": "Employment status adds risk on top of income, so an income-only cut-off misses it. Consider a stricter policy "
                   "(co-borrower, lower limit, collateral) for unemployed applicants rather than a flat income threshold.",
        "numbers": {"unemployed_rate": float(un["default_rate"]), "unemployed_n": int(un["accounts"]),
                    "low_income_unemployed": float(un_lo.mean()), "low_income_salaried": float(sal_lo.mean()),
                    "income_lt40k_rate": float(inc.loc["<40k", "default_rate"]), "income_100k_rate": float(inc.loc["100k+", "default_rate"])}})

    # 3. Delinquency ladder (early warning) --------------------------------------------------------------
    ladder = df.assign(b=pd.cut(df["rep_delinquent"], [-1, 0, 1, 3, 99], labels=["0", "1", "2-3", "4+"]))
    lg = ladder.groupby("b", observed=True)[C.TARGET].agg(["mean", "size"])
    early = df.assign(e=np.where(df["rep_delinquent_early"] >= 2, "2+ late/missed in months 1-6", "0-1 in months 1-6")) \
              .groupby("e")[C.TARGET].agg(["mean", "size"])
    out.append({
        "id": "delinquency_ladder", "cross_dimensional": False, "dimension": "repayment behaviour",
        "headline": f"Delinquency is a steep early-warning ladder: {_pct(lg.loc['0', 'mean'], 0)} -> {_pct(lg.loc['4+', 'mean'], 0)} default",
        "evidence": (f"0 late/missed months: {_pct(lg.loc['0', 'mean'])} (n={int(lg.loc['0', 'size']):,}); 1: {_pct(lg.loc['1', 'mean'])}; "
                     f"2-3: {_pct(lg.loc['2-3', 'mean'])}; 4+: {_pct(lg.loc['4+', 'mean'])} (n={int(lg.loc['4+', 'size']):,}, "
                     f"{_pct(lg.loc['4+', 'size'] / len(df))} of accounts). Already by months 1-6, 2+ delinquent months means "
                     f"{_pct(early.loc['2+ late/missed in months 1-6', 'mean'])} vs {_pct(early.loc['0-1 in months 1-6', 'mean'])}."),
        "so_what": "Risk is visible within six months of origination. Trigger collections outreach after the second delinquent month "
                   "instead of waiting for deep arrears; this is also the basis of the dashboard's anomaly review queue.",
        "numbers": {"zero_delq_rate": float(lg.loc["0", "mean"]), "four_plus_rate": float(lg.loc["4+", "mean"]),
                    "four_plus_share": float(lg.loc["4+", "size"] / len(df))}})

    # 4. Bureau write-offs x loan type (cross-dimensional) -------------------------------------------------
    r2, n2 = cross_table(df, "loan_type", "bureau_profile", min_n=100)
    bp = segment_table(df, "bureau_profile").set_index("bureau_profile")
    wo, nw = bp.loc["Has written-off line(s)"], bp.loc["No write-offs"]
    gap = (r2["Has written-off line(s)"] - r2["No write-offs"]).dropna().sort_values(ascending=False)
    top_t = gap.index[0]
    out.append({
        "id": "bureau_writeoff", "cross_dimensional": True, "dimension": "loan type x bureau profile",
        "headline": f"A written-off line elsewhere lifts default from {_pct(nw['default_rate'])} to {_pct(wo['default_rate'])}, most on {top_t}s",
        "evidence": (f"{_pct(wo['account_share'], 0)} of accounts (n={int(wo['accounts']):,}) belong to customers with a written-off bureau line. "
                     f"Largest gap on {top_t}: {_pct(r2.loc[top_t, 'Has written-off line(s)'])} vs {_pct(r2.loc[top_t, 'No write-offs'])}. "
                     f"Thin-file customers: {_pct(bp.loc['Thin file (no bureau lines)', 'default_rate'])} (n={int(bp.loc['Thin file (no bureau lines)', 'accounts']):,})."),
        "so_what": "External write-offs are a cheap, reliable screen that should tighten limits or pricing, especially on unsecured "
                   "business and card lending; thin-file customers sit above average and deserve manual review rather than a default 'clean' assumption.",
        "numbers": {"writeoff_rate": float(wo["default_rate"]), "no_writeoff_rate": float(nw["default_rate"]), "top_loan_type": str(top_t)}})

    # 5. Geography is flat; age and income are not -------------------------------------------------------------
    reg, tier, ch = (segment_table(df, d)["default_rate"] for d in ("region", "city_tier", "disbursal_channel"))
    age = segment_table(df, "age_band").set_index("age_band")
    spread = lambda s: (s.max() - s.min())
    out.append({
        "id": "geography_vs_age", "cross_dimensional": False, "dimension": "region / channel vs age",
        "headline": "Geography and channel barely move risk; age and income do",
        "evidence": (f"Region range {_pct(reg.min())}-{_pct(reg.max())}, city tier {_pct(tier.min())}-{_pct(tier.max())}, channel "
                     f"{_pct(ch.min())}-{_pct(ch.max())}. By age: <30 defaults at {_pct(age.loc['<30', 'default_rate'])} (n={int(age.loc['<30', 'accounts']):,}) "
                     f"vs {_pct(age.loc['40-49', 'default_rate'])} for 40-49 and {_pct(age.loc['60+', 'default_rate'])} for 60+ (n={int(age.loc['60+', 'accounts'])})."),
        "so_what": "Regional or channel tightening would cost volume without reducing risk. Focus policy on borrower attributes (income, "
                   "employment, bureau history) and product mix. Treat age as a monitoring variable only, subject to fair-lending review.",
        "numbers": {"region_spread_pp": float(100 * spread(reg)), "channel_spread_pp": float(100 * spread(ch)),
                    "under30_rate": float(age.loc["<30", "default_rate"])}})
    return out
