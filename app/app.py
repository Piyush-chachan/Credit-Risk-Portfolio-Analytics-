"""
Credit Risk Portfolio Analytics - Streamlit dashboard.

Run with:   streamlit run app/app.py

How this file is organised (read it top to bottom):
    1. Setup            - logging, page settings, colours
    2. Loading data     - small cached functions that read the files made by `python -m src.build_assets`
    3. Small helpers    - money formatting, the info box used by both AI features
    4. Sidebar filters  - the filters that apply to most tabs
    5. One function per tab
    6. main()           - loads data, draws the sidebar, then calls each tab function

IMPORTANT about the logs: Streamlit re-runs this whole file every time you click something.
So you will see "[RERUN]" in the terminal on every click. That is normal.
Messages inside the cached loaders only appear the FIRST time (cache miss), which is how you can
tell whether data came from the cache or was read from disk.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

print("APP FILE:", Path(__file__).resolve())
print("PROJECT ROOT:", ROOT)
print("ROOT EXISTS:", ROOT.exists())
print("SRC EXISTS:", (ROOT / "src").is_dir())
print("SRC INIT EXISTS:", (ROOT / "src" / "__init__.py").exists())

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


print("PYTHON PATH:", sys.path[0])

import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import analytics, anomaly, assets, model
from src import config as C
from src.ask_portfolio import PortfolioAssistant
from src.ingestion import find_file
from src.llm import get_llm



# =====================================================================================
# 1. SETUP
# =====================================================================================
# Print log lines to the terminal (and to the Streamlit Cloud "Manage app" logs).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("dashboard")

st.set_page_config(page_title="Credit Risk Portfolio Analytics",
                   page_icon=":bar_chart:", layout="wide")

# colours used for the four risk segments
SEGMENT_COLOURS = {"Low": "#2e8b57", "Medium": "#e0b000",
                   "High": "#e8751a", "Very High": "#c0392b"}
# colours used for the repayment status bars
STATUS_COLOURS = {"On-Time": "#2e8b57", "Late": "#e0b000", "Missed": "#c0392b"}

# a little CSS to tidy spacing
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem}
    div[data-testid="stMetricValue"] {font-size: 1.5rem}
    .card {border: 1px solid rgba(128,128,128,.35); border-radius: 10px; padding: .8rem 1rem; margin-bottom: .7rem}
    .tag {display:inline-block; padding:1px 8px; border-radius:9px; font-size:.72rem; background:rgba(80,120,200,.18); margin-left:6px}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================================
# 2. LOADING DATA  (each function runs once, then Streamlit remembers the result)
# =====================================================================================
@st.cache_resource(show_spinner="First start: building dataset, model and review queue (about 1 minute) ...")
def make_sure_files_exist():
    """If the files from build_assets are missing, build them now."""
    logger.info("[BOOT] Checking that processed files exist ...")
    if assets.artifacts_ready():
        logger.info("[BOOT] Files already exist. Nothing to build.")
    else:
        logger.info(
            "[BOOT] Files missing. Building them now (this takes about a minute) ...")
        assets.ensure_built()
        logger.info("[BOOT] Build finished.")
    return True


@st.cache_data(show_spinner=False)
def load_portfolio():
    logger.info(
        "[LOAD] Reading portfolio (one row per loan account) from disk ...")
    portfolio = assets.load_portfolio()
    logger.info(
        f"[LOAD] Portfolio loaded: {len(portfolio):,} rows, {portfolio.shape[1]} columns")
    return portfolio


@st.cache_data(show_spinner=False)
def load_collections():
    logger.info("[LOAD] Reading collections events from disk ...")
    collections = assets.load_csv("collections")
    # origination_year can come back as '2021.0' - clean it to '2021'
    collections["origination_year"] = collections["origination_year"].astype(
        str).str.replace(r"\.0$", "", regex=True)
    return collections


@st.cache_data(show_spinner=False)
def load_results():
    """All the small result files (metrics, insights, importance, etc.) in one dictionary."""
    logger.info(
        "[LOAD] Reading model results, insights, anomaly queue and data-quality register ...")
    results = {}
    results["metrics"] = assets.load_json("metrics")
    results["curves"] = assets.load_json("curves")
    results["insights"] = assets.load_json("insights")
    results["meta"] = assets.load_json("meta")
    results["dq"] = pd.DataFrame(assets.load_json("dq"))
    results["importance"] = assets.load_csv("importance")
    results["deciles"] = assets.load_csv("deciles")
    results["psi"] = assets.load_csv("psi")
    results["ablation"] = assets.load_csv("ablation")
    results["drivers"] = assets.load_csv("drivers").set_index("account_id")
    results["anomalies"] = assets.load_csv("anomalies")
    logger.info("[LOAD] Results loaded.")
    return results


@st.cache_data(show_spinner=False)
def load_rows_from_raw_table(table_name, key_column, key_value):
    """Get the raw rows for ONE account or customer, e.g. its 12 repayment months."""
    logger.info(
        f"[LOAD] Reading raw table '{table_name}' for {key_column} = {key_value}")
    table = pd.read_csv(find_file(table_name))
    rows = table[table[key_column] == key_value]
    return rows.reset_index(drop=True)


# =====================================================================================
# 3. SMALL HELPERS
# =====================================================================================
def format_money(amount):
    """1234567 -> '$1.2M',  45000 -> '$45,000'"""
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}M"
    return f"${amount:,.0f}"


def show_info_box(title, trigger, false_positive, false_negative, human_step):
    """The 'what triggers it / false positive / false negative / human step' box for each AI feature."""
    with st.expander(title, expanded=False):
        st.markdown(f"**What triggers it.** {trigger}")
        st.markdown(f"**False positive looks like.** {false_positive}")
        st.markdown(f"**False negative looks like.** {false_negative}")
        st.markdown(f"**Human-in-the-loop.** {human_step}")


def repayment_chart(account_id, title):
    """Bar chart of days-past-due for each of the 12 months of one account."""
    repayments = load_rows_from_raw_table(
        "repayment_history", "account_id", account_id)
    figure = px.bar(
        repayments, x="payment_month", y="days_past_due",
        color="payment_status", color_discrete_map=STATUS_COLOURS, title=title,
    )
    figure.update_layout(height=260, margin=dict(t=40, b=10))
    return figure


# =====================================================================================
# 4. SIDEBAR FILTERS
# =====================================================================================
def multiselect_filter(label, column, all_data, fixed_order=None):
    """A multiselect that starts with every value selected."""
    if fixed_order is not None:
        options = fixed_order
    else:
        options = sorted(all_data[column].dropna().astype(str).unique())
    return st.multiselect(label, options, default=options)


def draw_sidebar_and_filter(all_data, llm):
    """Draw the sidebar widgets and return only the rows that match the choices."""
    with st.sidebar:
        st.header("Portfolio filters")
        st.caption("These apply to the Portfolio, Account and Anomaly tabs. "
                   "Ask the Portfolio always uses the full portfolio.")

        regions = multiselect_filter("Region", "region", all_data)
        loan_types = multiselect_filter("Loan type", "loan_type", all_data)
        employment_types = multiselect_filter(
            "Employment type", "employment_type", all_data)
        income_bands = multiselect_filter(
            "Income band", "income_band", all_data, analytics.ORDERS["income_band"])
        age_bands = multiselect_filter(
            "Age band", "age_band", all_data, analytics.ORDERS["age_band"])
        risk_segments = multiselect_filter(
            "Model risk segment", "risk_segment", all_data, C.RISK_SEGMENT_LABELS)
        years = multiselect_filter(
            "Origination year", "origination_year", all_data)

        st.divider()
        if llm is not None:
            st.caption(
                f"LLM: **connected** ({llm.client.base_url} / {llm.model})")
        else:
            st.caption(
                "LLM: **no API key set** - Ask the Portfolio runs in offline rule-based mode.")
        st.caption("Synthetic data. Numbers describe this snapshot only.")

    # keep a row only if it passes every filter
    keep = (
        all_data["region"].isin(regions)
        & all_data["loan_type"].isin(loan_types)
        & all_data["employment_type"].isin(employment_types)
        & all_data["income_band"].isin(income_bands)
        & all_data["age_band"].isin(age_bands)
        & all_data["risk_segment"].isin(risk_segments)
        & all_data["origination_year"].isin(years)
    )
    filtered = all_data[keep]
    logger.info(
        f"[FILTER] {len(filtered):,} of {len(all_data):,} accounts match the sidebar filters")
    return filtered


# =====================================================================================
# 5a. TAB: PORTFOLIO
# =====================================================================================
def show_kpis(filtered):
    kpis = analytics.kpis(filtered)
    columns = st.columns(6)
    columns[0].metric("Loan accounts", f"{kpis['accounts']:,}")
    columns[1].metric("Customers", f"{kpis['customers']:,}")
    columns[2].metric("Exposure (orig. amount)",
                      format_money(kpis["exposure"]))
    columns[3].metric(
        "Default rate", f"{kpis['default_rate']:.1%}", help="Defaulted accounts / accounts")
    columns[4].metric("Exposure-weighted default rate", f"{kpis['exposure_default_rate']:.1%}",
                      help="Defaulted loan amount / total loan amount")
    columns[5].metric("Avg model PD", f"{kpis['avg_pd']:.1%}")
    return kpis


def show_risk_segments(filtered):
    st.subheader("Risk segments (model PD quantile bands)")
    segments = analytics.segment_table(filtered, "risk_segment")
    names = segments["risk_segment"].astype(str)
    bar_colours = [SEGMENT_COLOURS[name] for name in names]

    left, right = st.columns([3, 2])
    with left:
        figure = go.Figure()
        figure.add_bar(x=names, y=segments["default_rate"], name="Actual default rate",
                       marker_color=bar_colours, text=[f"{v:.1%}" for v in segments["default_rate"]])
        figure.add_scatter(x=names, y=segments["avg_pd"], name="Predicted PD", mode="markers",
                           marker=dict(symbol="diamond", size=12, color="black"))
        figure.update_layout(height=320, margin=dict(t=10, b=10), yaxis_tickformat=".0%",
                             legend=dict(orientation="h", y=1.12))
        st.plotly_chart(figure, use_container_width=True)
    with right:
        table = pd.DataFrame({
            "risk_segment": names,
            "accounts": segments["accounts"],
            "exposure": segments["exposure"],
            # show as a percent number
            "exposure_share": segments["exposure_share"] * 100,
            "default_rate": segments["default_rate"] * 100,
        })
        st.dataframe(table, hide_index=True, use_container_width=True, column_config={
            "exposure": st.column_config.NumberColumn("Exposure", format="$%.0f"),
            "exposure_share": st.column_config.NumberColumn("Exposure share", format="%.1f%%"),
            "default_rate": st.column_config.NumberColumn("Default rate", format="%.1f%%"),
        })
        st.caption(
            "Low = bottom 40% of predicted PD, Medium = 40-70%, High = 70-90%, Very High = top 10%.")


def show_dimension_charts(filtered, portfolio_default_rate):
    st.subheader("Trends across dimensions and drill-down")
    dimension_names = list(analytics.DIMENSIONS)
    dimension = st.selectbox("Dimension", dimension_names,
                             format_func=analytics.DIMENSIONS.get, index=0)
    table = analytics.segment_table(filtered, dimension)

    left, right = st.columns(2)
    with left:
        label = analytics.DIMENSIONS[dimension].lower()
        figure = px.bar(
            table, x=dimension, y="default_rate",
            text=table["default_rate"].map("{:.1%}".format),
            color="default_rate", color_continuous_scale="OrRd",
            # error bar going up
            error_y=table["ci_high"] - table["default_rate"],
            error_y_minus=table["default_rate"] -
            table["ci_low"],     # error bar going down
            hover_data={"accounts": True, "exposure": ":,.0f"},
            title=f"Default rate by {label} (95% confidence interval)",
        )
        figure.add_hline(y=portfolio_default_rate,
                         line_dash="dot", annotation_text="portfolio")
        figure.update_layout(height=340, margin=dict(
            t=40, b=10), yaxis_tickformat=".0%", coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = px.bar(table, x=dimension, y="exposure_share",
                        text=table["exposure_share"].map("{:.0%}".format), title="Share of exposure")
        figure.add_scatter(x=table[dimension].astype(str), y=table["account_share"],
                           mode="markers+lines", name="share of accounts")
        figure.update_layout(height=340, margin=dict(t=40, b=10), yaxis_tickformat=".0%",
                             legend=dict(orientation="h", y=1.12))
        st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Error bars are 95% confidence intervals. A wide bar means a small group, so be careful reading it.")
    return dimension, table


def show_drilldown(filtered, dimension, table):
    st.markdown(
        "**Drill down**: pick one value to list its accounts, riskiest first")
    values = table[dimension].astype(str).tolist()
    chosen = st.selectbox(
        f"{analytics.DIMENSIONS[dimension]} value", values, key="drill_value")
    logger.info(f"[PORTFOLIO] Drill-down on {dimension} = {chosen}")

    accounts = filtered[filtered[dimension].astype(str) == chosen]
    accounts = accounts.sort_values("pd_score", ascending=False)

    columns_to_show = ["account_id", "customer_id", "loan_type", "loan_amount", "annual_income",
                       "employment_type", "pd_score", "risk_segment", "rep_delinquent", "bur_max_dpd", C.TARGET]
    display = accounts[columns_to_show].head(200).copy()
    display["pd_score"] = display["pd_score"] * \
        100        # show as a percent number
    st.dataframe(display, hide_index=True, use_container_width=True, height=260, column_config={
        "pd_score": st.column_config.ProgressColumn("Model PD", min_value=0, max_value=80, format="%.1f%%"),
        "loan_amount": st.column_config.NumberColumn(format="$%.0f"),
        "annual_income": st.column_config.NumberColumn(format="$%.0f"),
    })
    st.caption(
        f"{len(accounts):,} accounts in '{chosen}'. Copy an account_id into the Account view tab.")


def show_heatmap(filtered):
    st.markdown("**Cross-dimensional cut**")
    dimension_names = list(analytics.DIMENSIONS)
    col1, col2, col3 = st.columns(3)
    row_dim = col1.selectbox("Rows", dimension_names, index=dimension_names.index("employment_type"),
                             format_func=analytics.DIMENSIONS.get)
    col_dim = col2.selectbox("Columns", dimension_names, index=dimension_names.index("loan_type"),
                             format_func=analytics.DIMENSIONS.get)
    min_accounts = col3.slider(
        "Hide cells with fewer accounts than", 10, 100, 30)

    if row_dim == col_dim:
        st.info("Pick two different dimensions.")
        return

    rate_table, count_table = analytics.cross_table(
        filtered, row_dim, col_dim, min_accounts)

    # build the text shown inside each cell: "25%  n=140"
    cell_text = []
    for row_label in rate_table.index:
        text_row = []
        for col_label in rate_table.columns:
            rate = rate_table.loc[row_label, col_label]
            count = count_table.loc[row_label, col_label]
            if pd.isna(rate):
                text_row.append("")
            else:
                text_row.append(f"{rate:.0%}<br>n={int(count)}")
        cell_text.append(text_row)

    figure = go.Figure(go.Heatmap(
        z=rate_table.values,
        x=[str(c) for c in rate_table.columns],
        y=[str(i) for i in rate_table.index],
        text=cell_text, texttemplate="%{text}",
        colorscale="OrRd", colorbar=dict(tickformat=".0%"),
    ))
    figure.update_layout(height=60 + 55 * len(rate_table),
                         margin=dict(t=10, b=10))
    st.plotly_chart(figure, use_container_width=True)


def show_vintage_trend(filtered):
    st.markdown(
        "**Vintage trend** (default rate by the quarter the loan was originated)")
    split_choice = st.selectbox("Split lines by", [
                                "(none)", "loan_type", "employment_type", "region", "risk_segment"])
    group_columns = ["origination_quarter"]
    if split_choice != "(none)":
        group_columns.append(split_choice)

    trend = filtered.groupby(group_columns).agg(
        accounts=("account_id", "size"),
        default_rate=(C.TARGET, "mean"),
    ).reset_index()

    line_colour = None if split_choice == "(none)" else split_choice
    figure = px.line(trend, x="origination_quarter", y="default_rate", color=line_colour,
                     markers=True, hover_data=["accounts"])
    figure.update_layout(height=300, margin=dict(
        t=10, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True)


def portfolio_tab(filtered):
    logger.info("[TAB] Portfolio tab started")
    kpis = show_kpis(filtered)
    show_risk_segments(filtered)
    dimension, table = show_dimension_charts(filtered, kpis["default_rate"])
    show_drilldown(filtered, dimension, table)
    show_heatmap(filtered)
    show_vintage_trend(filtered)
    logger.info("[TAB] Portfolio tab finished")


# =====================================================================================
# 5b. TAB: INSIGHTS
# =====================================================================================
def insights_tab(results):
    logger.info("[TAB] Insights tab started")
    st.subheader(
        "Portfolio insights (computed from the full data, with sample sizes)")
    number = 1
    for insight in results["insights"]:
        tag = ""
        if insight["cross_dimensional"]:
            tag = '<span class="tag">cross-dimensional</span>'
        st.markdown(
            f'<div class="card"><b>{number}. {insight["headline"]}</b>{tag}<br><small>{insight["dimension"]}</small></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Evidence.** {insight['evidence']}")
        st.markdown(f"**So what.** {insight['so_what']}")
        st.divider()
        number += 1


# =====================================================================================
# 5c. TAB: ACCOUNT VIEW
# =====================================================================================
def account_tab(filtered, all_data, results):
    logger.info("[TAB] Account view tab started")
    st.subheader("Account-level risk score and key drivers")

    # --- step 1: which account? either typed in, or picked from a list sorted by risk
    ids_by_risk = filtered.sort_values("pd_score", ascending=False)[
        "account_id"].tolist()
    typed_id = st.text_input(
        "Account ID (or pick from the list below, sorted by risk)", value="", placeholder="e.g. ACC200001")
    typed_id = typed_id.strip().upper()
    all_ids = set(all_data["account_id"])

    if typed_id in all_ids:
        account_id = typed_id
        logger.info(f"[ACCOUNT] Using typed account id {account_id}")
    else:
        if typed_id != "":
            st.warning(f"'{typed_id}' is not a known account ID.")
            logger.warning(
                f"[ACCOUNT] Typed id '{typed_id}' not found, falling back to the list")
        account_id = st.selectbox("Account", ids_by_risk, index=0)
        logger.info(f"[ACCOUNT] Using selected account id {account_id}")

    account = all_data[all_data["account_id"] == account_id].iloc[0]

    # --- step 2: headline numbers
    average_pd = all_data["pd_score"].mean()
    difference_pp = (account["pd_score"] - average_pd) * 100
    outcome = "Defaulted" if account[C.TARGET] == 1 else "Performing"

    columns = st.columns(5)
    columns[0].metric("Predicted PD", f"{account['pd_score']:.1%}",
                      delta=f"{difference_pp:+.1f} pp vs portfolio", delta_color="inverse")
    columns[1].metric("Risk segment", account["risk_segment"])
    columns[2].metric("Recorded outcome", outcome)
    columns[3].metric("Loan", account["loan_type"],
                      help=f"{format_money(account['loan_amount'])} over {int(account['tenure_months'])} months")
    columns[4].metric("Loan amount", format_money(account["loan_amount"]))

    # --- step 3: key drivers chart (left) and profile table (right)
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "**Key drivers of this score** (they add up to the score, relative to an average account)")
        drivers = results["drivers"].loc[account_id]
        biggest = drivers.abs().sort_values(ascending=False).head(
            10).index      # 10 biggest by size
        # reversed so biggest is on top
        top_drivers = drivers[biggest][::-1]
        bar_labels = [model.describe_driver(
            name, account) for name in top_drivers.index]
        bar_colours = ["#c0392b" if value >
                       0 else "#2e8b57" for value in top_drivers.values]
        figure = go.Figure(go.Bar(
            x=top_drivers.values, y=bar_labels, orientation="h", marker_color=bar_colours))
        figure.update_layout(height=380, margin=dict(
            t=10, b=10, l=10), xaxis_title="<- lowers risk | raises risk ->")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        st.markdown("**Profile**")
        if pd.isna(account["bur_avg_dpd"]):
            bureau_dpd_text = "n/a (thin file)"
        else:
            bureau_dpd_text = f"{account['bur_avg_dpd']:.1f}"
        profile = {
            "Customer": account["customer_id"],
            "Age / gender": f"{int(account['age'])} / {account['gender']}",
            "Employment": account["employment_type"],
            "Annual income": f"${account['annual_income']:,.0f}",
            "Region / tier": f"{account['region']} / {account['city_tier']}",
            "Interest rate": f"{account['interest_rate_pct']:.1f}%",
            "Secured": account["secured"],
            "Origination": str(account["origination_date"])[:10],
            "Bureau lines (active / written-off)":
                f"{int(account['bur_lines'])} ({int(account['bur_active'])} / {int(account['bur_writeoff'])})",
            "Bureau avg DPD": bureau_dpd_text,
            "Late / missed months (12m)": f"{int(account['rep_late'])} / {int(account['rep_missed'])}",
        }
        profile_table = pd.DataFrame({"": list(profile.keys()), "value": [
                                     str(v) for v in profile.values()]})
        st.dataframe(profile_table, hide_index=True, use_container_width=True)

    # --- step 4: repayment history
    st.plotly_chart(repayment_chart(
        account_id, "Repayment history: days past due by month"), use_container_width=True)

    # --- step 5: is this account in the anomaly queue?
    queue = results["anomalies"]
    flag = queue[queue["account_id"] == account_id]
    if len(flag) > 0:
        first_flag = flag.iloc[0]
        st.error(f"Flagged in the anomaly queue ({first_flag['severity']}): "
                 + first_flag["reasons"].replace(" | ", "  \n- "))
        logger.info(f"[ACCOUNT] {account_id} is in the anomaly queue")

    # --- step 6: raw bureau and collections rows
    with st.expander("Bureau lines and collections events for this customer / account"):
        bureau_rows = load_rows_from_raw_table(
            "bureau_credit", "customer_id", account["customer_id"])
        collection_rows = load_rows_from_raw_table(
            "collections_events", "account_id", account_id)
        st.dataframe(bureau_rows, hide_index=True, use_container_width=True)
        st.dataframe(collection_rows, hide_index=True,
                     use_container_width=True)
        st.caption(
            "Collections are shown for context only. They are never used by the model.")


# =====================================================================================
# 5d. TAB: DEFAULT MODEL
# =====================================================================================
def model_tab(results):
    logger.info("[TAB] Default model tab started")
    metrics = results["metrics"]
    logistic = metrics["Logistic Regression"]
    lightgbm = metrics["LightGBM"]

    st.subheader("Default model: logistic regression baseline vs LightGBM")
    st.caption(f"{metrics['cv']}. {metrics['n_accounts']:,} accounts / {metrics['n_customers']:,} customers, "
               f"base default rate {metrics['default_rate']:.1%}.")

    # --- comparison table
    table_rows = []
    for model_name in ["Logistic Regression", "LightGBM"]:
        cv_scores = metrics[model_name]
        time_scores = metrics["out_of_time"][model_name]
        label = model_name
        if model_name == metrics["champion"]:
            label = model_name + "  (champion)"
        table_rows.append({
            "Model": label,
            "AUC (CV)": cv_scores["AUC"],
            "AUC 95% CI": f"{cv_scores['AUC_CI95'][0]:.3f}-{cv_scores['AUC_CI95'][1]:.3f}",
            "KS (CV)": cv_scores["KS"],
            "Gini (CV)": cv_scores["Gini"],
            "AUC out-of-time": time_scores["AUC"],
            "KS out-of-time": time_scores["KS"],
            "Gini out-of-time": time_scores["Gini"],
        })
    number_columns = ["AUC (CV)", "KS (CV)", "Gini (CV)",
                      "AUC out-of-time", "KS out-of-time", "Gini out-of-time"]
    column_config = {name: st.column_config.NumberColumn(
        format="%.3f") for name in number_columns}
    st.dataframe(pd.DataFrame(table_rows), hide_index=True,
                 use_container_width=True, column_config=column_config)
    st.caption(f"Out-of-time: {metrics['out_of_time']['split']}. {metrics['selection_rule']}. "
               "The ML model did not clear that bar, so the simpler logistic model is the champion.")
    st.info("Honest reading: an AUC of about 0.68 is modest and close to the most this synthetic data allows. "
            "Use the model to rank and monitor risk, not to make automatic decisions.")

    # --- ROC curve (left) and calibration by decile (right)
    deciles = results["deciles"]
    curves = results["curves"]
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        figure.add_scatter(x=curves["roc_lr"]["fpr"], y=curves["roc_lr"]
                           ["tpr"], name=f"Logistic (AUC {logistic['AUC']:.3f})")
        figure.add_scatter(x=curves["roc_gb"]["fpr"], y=curves["roc_gb"]
                           ["tpr"], name=f"LightGBM (AUC {lightgbm['AUC']:.3f})")
        figure.add_scatter(x=[0, 1], y=[0, 1], name="random",
                           line=dict(dash="dot", color="grey"))
        figure.update_layout(title="ROC curve (out-of-fold)", height=330, margin=dict(t=40, b=10),
                             xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = go.Figure()
        figure.add_bar(
            x=deciles["decile"], y=deciles["actual_rate"], name="Actual default rate")
        figure.add_scatter(x=deciles["decile"], y=deciles["avg_pd"], name="Predicted PD",
                           mode="lines+markers", line=dict(color="black"))
        figure.update_layout(title="Calibration by risk decile (1 = riskiest)", height=330,
                             margin=dict(t=40, b=10), yaxis_tickformat=".0%", xaxis_title="Decile")
        st.plotly_chart(figure, use_container_width=True)

    st.dataframe(deciles, hide_index=True, use_container_width=True, column_config={
        "avg_pd": st.column_config.NumberColumn("Avg PD", format="%.3f"),
        "actual_rate": st.column_config.NumberColumn("Actual rate", format="%.3f"),
        "lift": st.column_config.NumberColumn(format="%.2f"),
        "cum_bad_capture": st.column_config.NumberColumn("Cum. defaulters captured", format="%.3f"),
        "cum_good_capture": st.column_config.NumberColumn(format="%.3f"),
        "KS_at_decile": st.column_config.NumberColumn("KS at decile", format="%.3f"),
    })
    best_ks = deciles["KS_at_decile"].max()
    top_two_deciles = deciles.loc[1, "cum_bad_capture"]
    st.caption(f"KS = biggest gap between cumulative defaulters and non-defaulters = {best_ks:.3f}. "
               f"The riskiest 20% of accounts contain {top_two_deciles:.0%} of all defaulters.")

    # --- feature importance
    st.markdown("**Feature importance**")
    importance = results["importance"]
    left, right = st.columns(2)
    with left:
        top_logistic = importance.head(15).iloc[::-1]
        figure = px.bar(top_logistic, x="mean_abs_driver", y="label", orientation="h",
                        title="Champion (logistic): average impact on the score")
        figure.update_layout(height=420, margin=dict(
            t=40, b=10), yaxis_title="")
        st.plotly_chart(figure, use_container_width=True)
    with right:
        top_shap = importance.sort_values(
            "mean_abs_shap_gbm", ascending=False).head(15).iloc[::-1]
        figure = px.bar(top_shap, x="mean_abs_shap_gbm", y="label", orientation="h",
                        title="LightGBM: average |SHAP| value (out-of-fold)")
        figure.update_layout(height=420, margin=dict(
            t=40, b=10), yaxis_title="")
        st.plotly_chart(figure, use_container_width=True)
    st.dataframe(importance[["label", "lr_direction", "mean_abs_driver", "mean_abs_shap_gbm"]],
                 hide_index=True, use_container_width=True, height=220)

    # --- what each group of features adds + leakage checks
    st.markdown("**What each feature group adds, and the leakage checks**")
    ablation = results["ablation"]
    st.dataframe(ablation, hide_index=True, use_container_width=True, column_config={
        "AUC_LR": st.column_config.NumberColumn(format="%.4f"),
        "KS_LR": st.column_config.NumberColumn(format="%.4f"),
    })
    auc_origination_only = ablation.iloc[0].AUC_LR
    auc_final = ablation.iloc[2].AUC_LR
    auc_with_collections = ablation.iloc[4].AUC_LR
    st.markdown(
        f"- Loan and customer data alone give AUC {auc_origination_only:.3f}. "
        f"Adding bureau history and 12 months of repayment lifts it to {auc_final:.3f}.\n"
        f"- **Early-window test:** using only repayment months 1-6 still gives {metrics['early_window_auc_lr']:.3f}, "
        "so the signal is not just a late-stage effect of default itself.\n"
        f"- **Collections left out:** adding them moves AUC from {auc_final:.4f} to {auc_with_collections:.4f} (no gain). "
        "They are recorded after an account is already delinquent.\n"
        "- Gender and marital status are not used as inputs. Bureau data is a current snapshot "
        "(62% of lines were opened after the loan), so this is a monitoring score, not an application score."
    )

    # --- monitoring
    st.markdown("**Post-deployment monitoring**")
    st.dataframe(results["psi"], hide_index=True, use_container_width=True, height=240,
                 column_config={"PSI": st.column_config.NumberColumn(format="%.4f")})
    st.markdown(
        "Demo: PSI compares loans originated up to 2021 against later ones. Below 0.10 is stable, 0.10-0.25 watch, above 0.25 act.\n\n"
        "In production, check monthly: (1) data drift - PSI per feature and missing rates; (2) score drift - PSI of the PD and "
        "share per risk segment; (3) accuracy - AUC/KS on matured loans, alert if AUC drops by more than 0.03 below the validation "
        "range; (4) calibration - predicted vs actual default rate per segment; (5) fairness by protected group; "
        "(6) anomaly confirm/dismiss rates. Retrain on a breach and compare old vs new before switching."
    )


# =====================================================================================
# 5e. TAB: ANOMALY REVIEW
# =====================================================================================
def anomaly_tab(filtered, all_data, results):
    logger.info("[TAB] Anomaly review tab started")
    st.subheader("Anomaly review queue")

    show_info_box(
        "AI feature 2: trigger, false positive, false negative, human-in-the-loop",
        trigger="A batch job checks every account with four detectors and queues those scoring 3 or more: "
                "(A) last 3 months vs the account's own previous 9, (B) a metric in the top 2% of its peer group "
                "(loan type x income tier), (C) bureau vs internal behaviour disagreeing, (D) an IsolationForest fitted inside each peer group.",
        false_positive="An account is flagged but is fine: a one-off late payment after a clean record, "
                       "or a high loan-to-income that is normal for a large mortgage.",
        false_negative="An account is getting worse but is not flagged: a slow drift that never crosses a threshold, "
                       "or a customer with no bureau history to compare against.",
        human_step="The analyst reads the plain-English reasons, checks the repayment history, then records Confirmed / "
                   "Dismissed / Needs more info with a note. Every decision is saved to an audit log with time and reviewer. "
                   "Nothing happens automatically, and a flag is not a prediction of default.",
    )

    lift = results["meta"]["anomaly_lift"]
    st.caption(f"Queue: {lift['flagged_accounts']:,} accounts ({lift['flag_rate']:.1%}). Check only (the detector never sees the label): "
               f"flagged accounts default at {lift['default_rate_flagged']:.1%} vs {lift['default_rate_unflagged']:.1%} "
               f"for the rest ({lift['lift']:.2f}x).")

    # --- step 1: attach model score and review status to the queue
    score_columns = all_data[["account_id", "pd_score",
                              "risk_segment"]].drop_duplicates("account_id")
    queue = results["anomalies"].merge(
        score_columns, on="account_id", how="left")
    # respect the sidebar filters
    queue = queue[queue["account_id"].isin(filtered["account_id"])]

    decisions = anomaly.load_decisions()
    latest_status = anomaly.latest_status(decisions)
    queue["review_status"] = queue["account_id"].map(
        latest_status).fillna("Pending")
    logger.info(
        f"[ANOMALY] Queue has {len(queue):,} accounts; {len(decisions)} decisions saved so far")

    # --- step 2: filters above the table
    col1, col2, col3, col4 = st.columns(4)
    severities = col1.multiselect("Severity", ["High", "Medium", "Low"], default=[
                                  "High", "Medium", "Low"])
    detector_names = ["Own history", "Peer outlier",
                      "Bureau divergence", "Multivariate outlier"]
    detectors = col2.multiselect(
        "Detector fired", detector_names, default=detector_names)
    statuses = col3.multiselect("Review status", ["Pending", "Confirmed", "Dismissed", "Needs more info"],
                                default=["Pending", "Needs more info"])
    sort_column = col4.selectbox(
        "Sort by", ["priority", "anomaly_score", "loan_amount", "pd_score"])

    def any_selected_detector_fired(flag_types_text):
        for detector in detectors:
            if detector in flag_types_text:
                return True
        return False

    visible = queue[queue["severity"].isin(
        severities) & queue["review_status"].isin(statuses)]
    visible = visible[visible["flag_types"].apply(any_selected_detector_fired)]
    visible = visible.sort_values(sort_column, ascending=False)

    # --- step 3: counts and the table
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("In queue (after filters)", f"{len(visible):,}")
    m2.metric("Confirmed", int((latest_status == "Confirmed").sum()))
    m3.metric("Dismissed", int((latest_status == "Dismissed").sum()))
    m4.metric("Pending", int((queue["review_status"] == "Pending").sum()))

    table_columns = ["account_id", "severity", "anomaly_score", "flag_types", "loan_type", "peer_group",
                     "loan_amount", "pd_score", "review_status", "reasons"]
    display = visible[table_columns].head(500).copy()
    display["pd_score"] = display["pd_score"] * 100
    st.dataframe(display, hide_index=True, use_container_width=True, height=300, column_config={
        "pd_score": st.column_config.ProgressColumn("Model PD", min_value=0, max_value=80, format="%.1f%%"),
        "loan_amount": st.column_config.NumberColumn(format="$%.0f"),
        "anomaly_score": st.column_config.NumberColumn(format="%.1f"),
    })

    if visible.empty:
        st.success("Nothing left to review with these filters.")
    else:
        review_one_account(visible)

    # --- step 5: audit trail
    with st.expander("Decision log (append-only audit trail)"):
        st.dataframe(decisions.sort_values(
            "timestamp_utc", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Download decision log (CSV)", decisions.to_csv(
            index=False), "review_decisions.csv", "text/csv")
        st.caption(
            "On free hosting the file is wiped when the app restarts. A real system would use a database.")


def review_one_account(visible):
    """Show why one account was flagged and let the analyst record a decision."""
    account_id = st.selectbox(
        "Account to review", visible["account_id"].tolist(), key="anom_acct")
    row = visible[visible["account_id"] == account_id].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"**Why {account_id} was flagged** ({row['severity']}, score {row['anomaly_score']:.1f}; {row['flag_types']})")
        for reason in row["reasons"].split(" | "):
            st.markdown(f"- {reason}")
        st.caption(f"Model PD {row['pd_score']:.1%} ({row['risk_segment']}); peer group: {row['peer_group']}; "
                   f"loan {format_money(row['loan_amount'])}.")
    with right:
        figure = repayment_chart(account_id, "Repayment history (DPD)")
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, use_container_width=True)

    # a form means nothing is saved until the button is pressed
    with st.form("decision_form", clear_on_submit=True):
        col1, col2 = st.columns([1, 1])
        decision = col1.radio(
            "Decision", anomaly.DECISION_VALUES, horizontal=True)
        reviewer = col2.text_input("Reviewer", value="analyst")
        note = st.text_input("Note (why?)")
        save_pressed = st.form_submit_button("Save decision", type="primary")
        if save_pressed:
            anomaly.save_decision(account_id, decision, reviewer, note)
            logger.info(
                f"[ANOMALY] Decision saved: {account_id} -> {decision} (reviewer: {reviewer})")
            st.success(f"Saved: {account_id} -> {decision}")
            st.rerun()


# =====================================================================================
# 5f. TAB: ASK THE PORTFOLIO
# =====================================================================================
EXAMPLE_QUESTIONS = [
    "Which loan type has the highest default rate?",
    "Default rate by employment type and loan type",
    "Total exposure by region for Salaried customers",
    "Which risk segment holds the most exposure?",
    "How much was recovered by collections action type?",
    "What will the default rate be next year?",
]


def ask_tab(all_data, collections, llm):
    logger.info("[TAB] Ask the Portfolio tab started")
    st.subheader("Ask the Portfolio")

    show_info_box(
        "AI feature 1: trigger, false positive, false negative, human-in-the-loop",
        trigger="The analyst types a question. An LLM (one call) turns it into a structured query plan, or declines it. "
                "The plan is checked against allow-lists and run by DuckDB. The LLM never writes SQL and never calculates a number. "
                "What the LLM sees: the question, earlier turns and the list of metric/category names. Never customer rows.",
        false_positive="A valid question is declined, or answered with a different metric than intended "
                       "(for example 'risk' read as default rate). The plan, SQL and sample sizes are shown so this is easy to spot.",
        false_negative="A question the data cannot answer (a forecast, a cause, a city) gets a plausible answer. "
                       "Controls: scope check, allow-lists, filter values must exist in the data, and any LLM wording with a number "
                       "that is not in the result table is thrown away.",
        human_step="The analyst checks the plan, SQL and sample size before using a number, and can mark the answer Correct or Wrong "
                   "(saved). Groups with fewer than 30 rows are flagged.",
    )

    if llm is not None:
        mode_text = f"LLM mode ({llm.client}, {llm.model})"
    else:
        mode_text = "Offline rule-based mode (no API key: understands common question patterns only)"
    st.caption(
        f"Mode: **{mode_text}**. Answers use the full portfolio of {len(all_data):,} accounts.")

    # the assistant and chat history live in session_state so they survive Streamlit re-runs
    if "assistant" not in st.session_state:
        logger.info(
            "[ASK] Creating a new PortfolioAssistant for this browser session")
        st.session_state.assistant = PortfolioAssistant(
            all_data, collections, llm=llm)
        st.session_state.messages = []
    assistant = st.session_state.assistant

    # --- example question buttons
    clicked_example = None
    columns = st.columns(3)
    for position, example in enumerate(EXAMPLE_QUESTIONS):
        if columns[position % 3].button(example, key=f"example_{position}", use_container_width=True):
            clicked_example = example
    if st.button("Clear conversation"):
        logger.info("[ASK] Conversation cleared")
        assistant.reset()
        st.session_state.messages = []

    # --- show the conversation so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("table") is not None:
                st.dataframe(message["table"],
                             hide_index=True, use_container_width=True)
                for warning_text in message["warnings"]:
                    st.warning(warning_text)
                with st.expander("How this was computed (plan, SQL, source)"):
                    st.json(message["plan"])
                    st.code(message["sql"], language="sql")
                    st.caption(f"Source: {message['source']}  |  planner: {message['mode']}  |  "
                               f"LLM wording verified: {message['polished']}")

    # --- new question (typed or from an example button)
    typed_question = st.chat_input(
        "Ask about defaults, exposure, income, risk segments, collections ...")
    question = typed_question or clicked_example
    if question:
        logger.info(f"[ASK] New question: {question}")
        st.session_state.messages.append({"role": "user", "text": question})

        with st.spinner("Planning, validating and querying ..."):
            answer = assistant.ask(question)
        logger.info(
            f"[ASK] Result: status={answer.status}, stage={answer.stage or 'ok'}, mode={answer.mode}")

        if answer.status == "answered":
            reply = {"role": "assistant", "text": answer.text}
            table = answer.table.copy()
            for column in table.columns:
                if pd.api.types.is_float_dtype(table[column]):
                    table[column] = table[column].round(4)
            reply["table"] = table
            reply["plan"] = answer.plan
            reply["sql"] = answer.sql
            reply["source"] = answer.source
            reply["mode"] = answer.mode
            reply["polished"] = answer.polished
            reply["warnings"] = answer.warnings
        else:
            text = f"**Cannot answer.** {answer.text}"
            if answer.warnings:
                text = text + "\n\n" + "  \n".join(answer.warnings)
            reply = {"role": "assistant", "text": text}

        st.session_state.messages.append(reply)
        st.rerun()

    # --- Correct / Wrong feedback buttons after the latest answer
    messages = st.session_state.messages
    if len(messages) > 0 and messages[-1]["role"] == "assistant":
        feedback_columns = st.columns([1, 1, 6])
        for position, (label, code) in enumerate([("Correct", "ok"), ("Wrong", "bad")]):
            if feedback_columns[position].button(label, key=f"feedback_{code}"):
                question_text = messages[-2]["text"].replace('"', "")
                with open(C.STATE_DIR / "ask_feedback.csv", "a", encoding="utf-8") as file:
                    file.write(
                        f'{pd.Timestamp.utcnow().isoformat()},{code},"{question_text}"\n')
                logger.info(f"[ASK] Feedback saved: {code}")
                st.toast("Feedback recorded")


# =====================================================================================
# 5g. TAB: DATA QUALITY
# =====================================================================================
def data_quality_tab(results):
    logger.info("[TAB] Data quality tab started")
    st.subheader("Data ingestion, join logic and data-quality register")
    st.markdown(
        "**Grain and joins** (one row per loan account, 14,685 rows, checked after every join):\n\n"
        "`loan_accounts` is the base table. `customers` is joined on customer_id (many loans to one customer). "
        "Bureau lines are **summed up per customer first**, then joined on customer_id. The 12 monthly repayment rows are "
        "**summed up per account first**, then joined on account_id. Collections events are also summed per account "
        "(shown for context, never used by the model). Summing before joining stops rows multiplying: "
        "joining 176,220 raw repayment rows directly would repeat every loan twelve times."
    )
    register = results["dq"]
    only_issues = st.toggle("Show only checks that found issues", value=True)
    if only_issues:
        register = register[register["status"] == "Issue"]
    st.dataframe(register, hide_index=True, use_container_width=True, height=420,
                 column_config={"pct": st.column_config.NumberColumn("% affected", format="%.2f")})
    full_register = results["dq"]
    issue_count = int((full_register["status"] == "Issue").sum())
    st.caption(
        f"{issue_count} issues found in {len(full_register)} checks. Each decision is applied in src/ingestion.py.")


# =====================================================================================
# 6. MAIN  - this is the order things happen on every run
# =====================================================================================
def main():
    logger.info("[RERUN] ---------- Streamlit re-ran the script ----------")

    # step 1: make sure the data files exist, then load them
    make_sure_files_exist()
    all_data = load_portfolio()
    collections = load_collections()
    results = load_results()

    # step 2: connect to the LLM (returns None if no key is set)
    llm = get_llm()
    if llm is not None:
        logger.info(f"[LLM] Connected: {llm.model} / {llm.client.base_url}")
    else:
        logger.info(
            "[LLM] No API key found. Ask the Portfolio will use the offline rule-based mode.")

    # step 3: sidebar and filtered data
    filtered = draw_sidebar_and_filter(all_data, llm)

    # step 4: page title
    metrics = results["metrics"]
    logistic = metrics["Logistic Regression"]
    st.title("Credit Risk Portfolio Analytics")
    st.caption(f"{len(filtered):,} of {len(all_data):,} accounts selected  |  model: {metrics['champion']} "
               f"(AUC {logistic['AUC']:.3f}, KS {logistic['KS']:.3f}, Gini {logistic['Gini']:.3f})")
    if filtered.empty:
        logger.warning(
            "[FILTER] No accounts match the filters - stopping early")
        st.warning("No accounts match the current filters.")
        st.stop()

    # step 5: the tabs
    tab_names = ["Portfolio", "Insights", "Account view", "Default model",
                 "Anomaly review", "Ask the Portfolio", "Data quality"]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        portfolio_tab(filtered)
    with tabs[1]:
        insights_tab(results)
    with tabs[2]:
        account_tab(filtered, all_data, results)
    with tabs[3]:
        model_tab(results)
    with tabs[4]:
        anomaly_tab(filtered, all_data, results)
    with tabs[5]:
        ask_tab(all_data, collections, llm)
    with tabs[6]:
        data_quality_tab(results)
    logger.info("[RERUN] ---------- Finished drawing the page ----------")


main()
