"""Generate docs/architecture.pdf and docs/solution_overview.pdf (one page each) from the build artifacts.

Run after `python -m src.build_assets`:   python -m src.make_docs
All figures are read from data/processed so the documents always match the dashboard.
"""
from __future__ import annotations
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepInFrame

from . import config as C
from .assets import load_csv, load_json

DOCS = C.ROOT / "docs"
DOCS.mkdir(exist_ok=True)
GREEN, ORANGE, BLUE, GREY = colors.HexColor("#e6f4ea"), colors.HexColor("#fde8cf"), colors.HexColor("#dbe8fb"), colors.HexColor("#f1f1f1")
EDGE = colors.HexColor("#555555")


# --------------------------------------------------------------------------- architecture diagram
def architecture() -> None:
    m = load_json("metrics")
    W, H = landscape(A4)
    c = canvas.Canvas(str(DOCS / "architecture.pdf"), pagesize=(W, H))
    c.setTitle("Credit Risk Portfolio Analytics - Architecture")

    def box(x, y, w, h, title, lines, fill, tsize=8.6, lsize=6.9):
        c.setFillColor(fill); c.setStrokeColor(EDGE); c.setLineWidth(0.8)
        c.roundRect(x, y, w, h, 5, fill=1, stroke=1)
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", tsize)
        c.drawString(x + 6, y + h - 12, title)
        c.setFont("Helvetica", lsize)
        yy = y + h - 23
        for ln in lines:
            if ln.startswith("*"):
                c.setFont("Helvetica-Bold", lsize); ln = ln[1:]
            else:
                c.setFont("Helvetica", lsize)
            c.drawString(x + 6, yy, ln); yy -= lsize + 2.4

    def arrow(x1, y1, x2, y2, col=EDGE, label=None):
        c.setStrokeColor(col); c.setFillColor(col); c.setLineWidth(1.3)
        c.line(x1, y1, x2, y2)
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        p = c.beginPath()
        p.moveTo(x2, y2)
        p.lineTo(x2 - 6 * math.cos(a - .4), y2 - 6 * math.sin(a - .4))
        p.lineTo(x2 - 6 * math.cos(a + .4), y2 - 6 * math.sin(a + .4))
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        if label:
            c.setFont("Helvetica-Oblique", 6.5); c.setFillColor(col); c.drawString((x1 + x2) / 2 + 3, (y1 + y2) / 2 + 3, label)

    c.setFont("Helvetica-Bold", 14); c.setFillColor(colors.black)
    c.drawString(28, H - 30, "Credit Risk Portfolio Analytics - Architecture")
    c.setFont("Helvetica", 8)
    c.drawString(28, H - 43, "Raw data -> ingestion -> analytics -> model -> dashboard, with both AI features. Green = deterministic code, "
                 "orange = LLM call, blue = human step.")

    # ---- main pipeline row
    y, h, w, gap = H - 190, 128, 140, 21
    xs = [28 + i * (w + gap) for i in range(5)]
    box(xs[0], y, w, h, "1  RAW DATA", ["data/raw/*.csv (synthetic)", "customers  10,000", "loan_accounts  14,685 (target)", "bureau_credit  22,979",
                                         "repayment_history  176,220", "collections_events  8,635", "Read with: pandas"], GREY)
    box(xs[1], y, w, h, "2  INGESTION", ["src/ingestion.py - pandas", "Normalise + type-check", "*55 data-quality checks (register)", "Clean: recode, flag, impute",
                                         "Aggregate children FIRST:", " bureau->customer, repay->account", "Join + assert grain (14,685)"], GREEN)
    box(xs[2], y, w, h, "3  FEATURES + ANALYTICS", ["src/features.py, analytics.py - pandas", "35 model features:", " origination, bureau, 12m repayment",
                                                     "Segment profiles, cross-cuts,", "vintage trend, 5 insights", "Collections kept OUT of model"], GREEN)
    box(xs[3], y, w, h, "4  DEFAULT MODEL", ["src/model.py - scikit-learn, LightGBM", "Logistic (champion) vs LightGBM", "5-fold CV grouped by customer",
                                             f"AUC {m['Logistic Regression']['AUC']:.3f}  KS {m['Logistic Regression']['KS']:.3f}  Gini {m['Logistic Regression']['Gini']:.3f}",
                                             "Out-of-fold PD, risk segments,", "per-account drivers, PSI monitor"], GREEN)
    box(xs[4], y, w, h, "5  DASHBOARD", ["app/app.py - Streamlit + Plotly", "Hosted: Streamlit Community Cloud", "Portfolio | Insights | Account",
                                         "Model | Anomaly | Ask | Data quality", "Filters + drill-down", "Secrets: Streamlit secrets store"], BLUE)
    for i in range(4):
        arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    c.setFont("Helvetica", 6.5); c.setFillColor(EDGE)
    c.drawString(xs[1] + 4, y - 9, "-> analysis_ready.csv.gz")
    c.drawString(xs[3] + 4, y - 9, "-> scores, drivers, model.joblib")

    # ---- Anomaly lane
    ay, ah = H - 335, 118
    box(28, ay, 395, ah, "AI FEATURE 2  ANOMALY DETECTION   (no LLM: transparent rules + IsolationForest)",
        ["src/anomaly.py - pandas + scikit-learn.  Batch job over the analysis dataset; the target label is never used.",
         "*A Own history: last 3 months vs the account's previous 9 (sudden deterioration, new delinquency, streak)",
         "*B Peer outlier: metric in top 2% of loan type x income tier (n >= 100 per peer group)",
         "*C Bureau vs internal divergence: stressed elsewhere/clean with us, or the reverse",
         "*D Multivariate: IsolationForest fitted within each peer group",
         "Score >= 3 -> review queue with plain-English reasons and the numbers behind each.",
         "Output: ~5.5% of accounts; flagged accounts default 1.6x more often (validation only)."], GREEN)
    box(433, ay, 120, ah, "HUMAN STEP", ["Analyst reads reasons +", "repayment history, then", "*Confirm / Dismiss /", "*Needs more info + note.", "Append-only log (CSV):", "time, reviewer, decision.", "No automatic action."], BLUE)
    arrow(423, ay + ah / 2, 433, ay + ah / 2)
    arrow(xs[2] + w / 2, y, xs[2] + w / 2 - 60, ay + ah, label="features")
    arrow(553, ay + ah / 2, xs[4] + 30, y, label="queue + decisions")

    # ---- Ask lane
    qy, qh = 28, 190
    c.setFillColor(colors.black)
    box(28, qy, 805 - 28, qh, "AI FEATURE 1  ASK THE PORTFOLIO   (LLM: OpenAI / Azure OpenAI - key held in Streamlit secrets; DuckDB computes every number)", [], colors.white)
    step_w, sx, sy, sh = 142, 36, qy + 26, 112
    steps = [
        ("0 GUARD", GREEN, ["Deterministic.", "Empty / >400 chars /", "prompt-injection", "patterns -> canned", "decline."]),
        ("1 PLAN + SCOPE GATE", ORANGE, ["*LLM CALL 1 (temp 0, JSON)", "Sends: question, last 3", "turns (question + plan),", "catalogue of metric &", "dimension NAMES and", "category values.", "NO customer rows."]),
        ("2 VALIDATE", GREEN, ["Allow-listed metrics,", "dimensions, max 2", "group-bys; filter values", "must exist in data.", "Invalid -> rule-based", "planner or decline."]),
        ("3 EXECUTE", GREEN, ["DuckDB (in-memory).", "SQL built from fixed", "fragments only;", "values are BOUND", "parameters. The LLM", "never writes SQL."]),
        ("4 ANSWER + VERIFY", ORANGE, ["Deterministic text first.", "*LLM CALL 2 (optional):", "sends aggregated result", "(<= 15 rows) to reword.", "Kept only if every number", "matches the table."]),
    ]
    for i, (t, f, ls) in enumerate(steps):
        x = sx + i * (step_w + 10)
        box(x, sy, step_w, sh, t, ls, f, tsize=7.6, lsize=6.5)
        if i < 4:
            arrow(x + step_w, sy + sh / 2, x + step_w + 10, sy + sh / 2)
    c.setFont("Helvetica", 7); c.setFillColor(colors.black)
    c.drawString(36, sy + sh + 24, "Data: analysis-ready dataset + collections events, registered as in-memory DuckDB views (read-only).")
    c.drawString(36, sy + sh + 12, "Analyst question  ->  [0]  ->  [1]  ->  [2]  ->  [3]  ->  [4]  ->  answer + table + plan + SQL + source + small-sample warning (n < 30)")
    c.setFont("Helvetica", 6.8)
    c.drawString(36, qy + 10, "Fallbacks: no key or LLM error -> rule-based planner (offline mode, banner shown). Audit: every question, plan and outcome appended to data/state/ask_audit.jsonl. "
                 "Analyst can mark answers Correct / Wrong.")
    # Ask lane is embedded in the dashboard: question goes down, verified answer comes back up
    arrow(xs[4] + w - 50, y, xs[4] + w - 50, qy + qh, label="")
    arrow(xs[4] + w - 22, qy + qh, xs[4] + w - 22, y, label="")
    c.setFont("Helvetica-Oblique", 6.5); c.setFillColor(EDGE)
    c.drawRightString(xs[4] + w - 54, qy + qh + 60, "question")
    c.drawString(xs[4] + w - 18, qy + qh + 60, "answer")
    c.showPage(); c.save()


# --------------------------------------------------------------------------- solution overview
def overview() -> None:
    m = load_json("metrics"); ins = {i["id"]: i for i in load_json("insights")}; meta = load_json("meta")
    ab = load_csv("ablation"); dq = load_json("dq")
    lr, gb, oot = m["Logistic Regression"], m["LightGBM"], m["out_of_time"]
    lift = meta["anomaly_lift"]
    issues = sum(1 for r in dq if r["status"] == "Issue")

    H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=1)
    H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1f4e79"))
    B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.0, leading=9.7, spaceAfter=1.6)
    BL = ParagraphStyle("bl", parent=B, leftIndent=9, bulletIndent=0)
    SM = ParagraphStyle("sm", parent=B, fontSize=7.3, leading=8.6, textColor=colors.HexColor("#444444"))

    def bl(t): return Paragraph(t, BL, bulletText="\u2022")

    mort, emp, lad = ins["mortgage_concentration"]["numbers"], ins["employment_income"]["numbers"], ins["delinquency_ladder"]["numbers"]
    story_l = [
        Paragraph("What the dashboard shows and how an analyst uses it", H2),
        bl("<b>Portfolio</b> - KPIs (accounts, exposure, default rate, exposure-weighted default rate), model risk segments with actual vs predicted default, "
           "default rate by any of 17 dimensions with 95% intervals, <b>drill-down</b> to the accounts in a segment, a cross-dimensional heatmap and a vintage trend. "
           "Sidebar filters (region, loan type, employment, income, age, segment, year) apply throughout. Use: find where risk and dollars concentrate."),
        bl("<b>Insights</b> - five computed insights, each with evidence (n shown) and a 'so what'."),
        bl("<b>Account view</b> - predicted PD, segment, <b>key drivers</b> (log-odds contributions that sum to the score), profile, 12-month repayment chart, bureau lines, "
           "collections, and any anomaly flag. Use: explain a score to a credit officer."),
        bl("<b>Default model</b> - AUC/KS/Gini (CV and out-of-time), ROC, calibration by decile, importance (logistic and SHAP), feature-group ablation, PSI monitoring."),
        bl("<b>Anomaly review</b> and <b>Ask the Portfolio</b> - the two AI features (below). <b>Data quality</b> - join logic and the register of every issue and decision."),

        Paragraph("AI feature 1: Ask the Portfolio (LLM)", H2),
        Paragraph("<b>How it works.</b> One LLM call converts the question into a JSON query plan (or declines it). Python validates the plan against allow-lists and "
                  "DuckDB computes the figures with bound-parameter SQL. The answer text is built deterministically from the result; optional LLM wording is accepted "
                  "only if every number matches the table. Output: figures, n, filters, SQL and source. Unanswerable questions (forecasts, causes, city-level, individuals) are declined. "
                  "No key or an LLM error falls back to a rule-based planner. Only the question, history and category catalogue (and aggregated results) reach the LLM, never customer rows.", B),
        Paragraph("<b>Trigger</b> analyst submits a question. <b>False positive</b> a valid question declined, or answered with the wrong metric (shown via plan/SQL). "
                  "<b>False negative</b> an unanswerable question answered plausibly; blocked by scope gate, allow-lists and value checks. "
                  "<b>Human step</b> analyst checks plan, SQL and n; marks the answer Correct/Wrong (logged).", B),

        Paragraph("AI feature 2: Anomaly detection (explainable, no LLM)", H2),
        Paragraph("<b>How it works.</b> Four detectors: (A) last 3 months vs the account's own previous 9; (B) top 2% of its peer segment (loan type x income tier); "
                  "(C) bureau-vs-internal divergence; (D) IsolationForest within each peer segment. Score &gt;= 3 enters the queue with plain-English reasons. "
                  f"{lift['flagged_accounts']:,} accounts ({lift['flag_rate']:.1%}) are queued; flagged accounts default at {lift['default_rate_flagged']:.1%} vs "
                  f"{lift['default_rate_unflagged']:.1%} ({lift['lift']:.2f}x), though the label is never used.", B),
        Paragraph("<b>Trigger</b> batch score &gt;= 3. <b>False positive</b> a one-off delay or a legitimately high loan-to-income. <b>False negative</b> slow drift that never crosses a "
                  "threshold, or a thin-file customer with no bureau comparison. <b>Human step</b> analyst confirms / dismisses / requests info with a note; "
                  "append-only audit log; decisions drive threshold recalibration. Nothing is automated.", B),

        Paragraph("Key assumptions", H2),
        bl("<b>Exposure</b> = original loan_amount (no current balance on the loan table). Target <i>target_default</i> is taken as given; its horizon is not stated."),
        bl("<b>Bureau is a current snapshot</b> (62.5% of lines opened after the loan), so this is a behavioural/monitoring score, not an origination score."),
        bl("<b>Collections excluded</b> from the model (recorded after delinquency; 44.9% dated before origination). customer_since ignored (after the loan for 49.7%)."),
        bl("Gender and marital status are not model inputs. Synthetic data: results describe this snapshot, not a real bank."),
    ]
    tbl = Table([["Model", "AUC", "KS", "Gini", "OOT AUC"],
                 ["Logistic (champion)", f"{lr['AUC']:.3f}", f"{lr['KS']:.3f}", f"{lr['Gini']:.3f}", f"{oot['Logistic Regression']['AUC']:.3f}"],
                 ["LightGBM", f"{gb['AUC']:.3f}", f"{gb['KS']:.3f}", f"{gb['Gini']:.3f}", f"{oot['LightGBM']['AUC']:.3f}"]],
                colWidths=[88, 38, 38, 38, 50])
    tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BACKGROUND", (0, 0), (-1, 0), GREY),
                             ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    story_r = [
        Paragraph("Model results and approach", H2), tbl, Spacer(1, 2),
        Paragraph(f"<b>Target</b> target_default ({m['default_rate']:.1%} of {m['n_accounts']:,} accounts). <b>Features</b> {m['final_features_numeric']} numeric + "
                  f"{m['final_features_categorical']} categorical: origination (income, age, loan size, LTI, rate), bureau (write-offs, DPD, utilisation, thin-file) and 12-month "
                  "repayment (late/missed counts, streak, DPD, EMI burden). <b>Validation</b> 5-fold CV grouped by customer (a customer's loans never straddle train/test; "
                  f"AUC 95% CI {lr['AUC_CI95'][0]:.3f}-{lr['AUC_CI95'][1]:.3f}) plus an out-of-time test ({oot['split']}). No class re-weighting, so outputs are real PDs; "
                  "risk-segment PDs match realised rates. Champion rule: the ML model must beat the baseline by &gt; 0.005 AUC; it did not, so the interpretable logistic model wins. "
                  f"<b>Honest read:</b> AUC ~0.68 is modest and near this data's ceiling. Origination data alone gives {ab.iloc[0].AUC_LR:.3f}; the months 1-6 model still gives "
                  f"{m['early_window_auc_lr']:.3f} (no late-window artefact); collections add no measurable AUC ({ab.iloc[2].AUC_LR:.4f} -> {ab.iloc[4].AUC_LR:.4f}). Use as a ranking/monitoring aid.", B),
        Paragraph("<b>Monitoring.</b> Monthly PSI on score and features (&lt;0.10 stable, &gt;0.25 act), AUC/KS on matured cohorts (alert if AUC falls &gt; 0.03 below the CI), "
                  "predicted-vs-actual by segment, fairness by protected group, anomaly confirm/dismiss rates; retrain via champion-challenger.", B),

        Paragraph("Top 3 portfolio insights", H2),
        Paragraph(f"<b>1. Mortgages dominate dollar risk.</b> {mort['mortgage_account_share']:.1%} of accounts but {mort['mortgage_exposure_share']:.0%} of exposure; default "
                  f"{mort['mortgage_default_rate']:.1%} vs {m['default_rate']:.1%}; {mort['share_defaulted_exposure']:.0%} of defaulted exposure. The value-weighted default rate is "
                  f"{ins['mortgage_concentration']['evidence'].split('Value-weighted default rate is ')[1].split(' against')[0]} vs {m['default_rate']:.1%} by count. "
                  "<i>So what:</i> govern mortgage underwriting and early arrears first; report both rates.", B),
        Paragraph(f"<b>2. Unemployment compounds low income (cross-cut employment x income).</b> Unemployed default at {emp['unemployed_rate']:.0%} (n={emp['unemployed_n']}). "
                  f"Below 40k income: unemployed {emp['low_income_unemployed']:.1%} vs salaried {emp['low_income_salaried']:.1%}; income &lt;40k overall {emp['income_lt40k_rate']:.1%} vs "
                  f"{emp['income_100k_rate']:.1%} at 100k+. <i>So what:</i> an income-only cut-off misses employment risk; tighten terms for unemployed applicants.", B),
        Paragraph(f"<b>3. Delinquency is an early-warning ladder.</b> Default rises from {lad['zero_delq_rate']:.1%} (no late/missed month) to {lad['four_plus_rate']:.1%} (4+; "
                  f"{lad['four_plus_share']:.1%} of accounts), visible by month 6. <i>So what:</i> start collections outreach at the second delinquent month.", B),

        Paragraph("To go to production, we would need from the bank's IT and data teams", H2),
        bl("<b>Point-in-time data:</b> as-of-date bureau pulls, loan snapshots, a defined default definition and horizon, and event timestamps that pass integrity checks."),
        bl("<b>Access and platform:</b> read-only feeds from the core banking, bureau and collections systems on a daily/monthly schedule; a governed data lake; SSO and role-based access; "
           "approved hosting (private VPC) and an enterprise LLM endpoint (e.g. Azure OpenAI) with logging and no customer rows sent."),
        bl("<b>Governance:</b> model-risk validation, fair-lending review of age, a database for review decisions and audit logs, outcome labels fed back from collections, monitoring owners and thresholds."),
        Paragraph(f"<i>Data quality: {issues} issues found in {len(dq)} checks; all handled and logged in the dashboard (Data quality tab).</i>", SM),
    ]
    W, H = A4
    doc = BaseDocTemplate(str(DOCS / "solution_overview.pdf"), pagesize=A4, leftMargin=22, rightMargin=22, topMargin=22, bottomMargin=20,
                          title="Credit Risk Portfolio Analytics - Solution Overview")
    colw = (W - 44 - 12) / 2
    top = H - 22 - 30
    frames = [Frame(22, 20, colw, top - 20, 0, 0, 0, 0, id="L"), Frame(22 + colw + 12, 20, colw, top - 20, 0, 0, 0, 0, id="R")]

    def header(cv, d):
        cv.setFont("Helvetica-Bold", 13); cv.drawString(22, H - 36, "Credit Risk Portfolio Analytics - Solution Overview")
        cv.setFont("Helvetica", 7.4); cv.setFillColor(colors.HexColor("#444444"))
        cv.drawString(22, H - 48, "Streamlit dashboard with an LLM 'Ask the Portfolio' assistant and explainable anomaly detection, built on 5 synthetic bank tables (10,000 customers, 14,685 loans).")
    doc.addPageTemplates([PageTemplate(id="p", frames=frames, onPage=header)])
    doc.build([KeepInFrame(colw, top - 20, story_l, mode="shrink"), __import__("reportlab.platypus", fromlist=["FrameBreak"]).FrameBreak(),
               KeepInFrame(colw, top - 20, story_r, mode="shrink")])


if __name__ == "__main__":
    architecture()
    overview()
    print("Wrote docs/architecture.pdf and docs/solution_overview.pdf")
