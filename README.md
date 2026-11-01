# Credit Risk Portfolio Analytics

Streamlit dashboard over five synthetic bank tables (10,000 customers, 14,685 loans) with a default-risk model and two AI features:
**Ask the Portfolio** (LLM question answering grounded in the data) and an **explainable anomaly review queue**.

* Live app: `<paste your Streamlit Community Cloud URL here>`
* One-page docs: [`docs/architecture.pdf`](docs/architecture.pdf), [`docs/solution_overview.pdf`](docs/solution_overview.pdf)

## Run it (three commands, Python 3.12)

```bash
pip install -r requirements.txt            # 1. install (all direct dependencies pinned)
python -m src.build_assets                 # 2. clean data -> model -> anomaly queue -> insights (~35 s)
streamlit run app/app.py                   # 3. dashboard at http://localhost:8501
```

To enable the LLM, copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or the Azure OpenAI variables).
**Without a key the app still works**: Ask the Portfolio switches to a built-in rule-based planner and says so in a banner.
Tests: `pytest -q` (39 tests; no network, the LLM is faked).

Pre-built artifacts are committed in `data/processed/` and `data/models/` (about 3.5 MB), so the hosted app starts instantly;
if they are missing the app rebuilds them on first start (about a minute).

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo** (keep `data/raw/` and `data/processed/`).
2. share.streamlit.io -> New app -> repo, branch `main`, main file `app/app.py`; Advanced settings -> **Python 3.12**.
3. Advanced settings -> **Secrets**: paste `OPENAI_API_KEY = "sk-..."` (see `.streamlit/secrets.toml.example`). Never commit keys.
4. `packages.txt` installs `libgomp1` (LightGBM runtime). Put the app URL at the top of this README.

## What is in the dashboard
| Tab | Purpose |
|---|---|
| Portfolio | KPIs, model risk segments (actual vs predicted), default rate by 17 dimensions with 95% intervals, **drill-down** to accounts, cross-dimension heatmap, vintage trend; sidebar filters apply throughout |
| Insights | 5 computed insights with evidence (n) and a "so what" (2 are cross-dimensional) |
| Account view | PD, segment, **key drivers** (sum to the score), profile, 12-month repayment chart, bureau, collections, anomaly flag |
| Default model | AUC/KS/Gini (CV + out-of-time), ROC, calibration, logistic + SHAP importance, ablation, PSI monitoring plan |
| Anomaly review | Queue with plain-English reasons, accept/dismiss/needs-info with note, append-only audit log |
| Ask the Portfolio | Chat; plan, SQL, source and sample-size warnings shown with every answer |
| Data quality | Join logic and the register of every issue found and the decision taken |

## Design decisions (and why)
**Data.** Child tables are aggregated *before* joining (bureau to customer, repayment/collections to account) and the grain is asserted (14,685 rows) so joins cannot fan out. 55 checks run; 13 issues are logged with action and rationale, including four that a quick look misses:
49.7% of `customer_since` dates are after the loan; 44.9% of collections events pre-date origination; 62.5% of bureau lines were opened after the loan (so bureau is a *current snapshot*); repayment is exactly 12 months with no partial payments.

**Model.** Logistic regression (champion) vs LightGBM. Customer-grouped 5-fold CV (a customer's loans never straddle train and test), out-of-fold scores for every account, no class re-weighting (outputs are real PDs; segment PDs match realised rates), an out-of-time check, and an ablation. Collections and protected attributes are excluded.
Results: AUC 0.680 (95% CI 0.669-0.692), KS 0.270, Gini 0.360; out-of-time AUC 0.689. This is modest and close to what this data supports: richer features and boosting did not beat the baseline, so it is presented honestly as a ranking/monitoring aid.

**Ask the Portfolio.** One LLM call produces a JSON query plan (and acts as the scope gate); Python validates it against allow-lists; DuckDB runs SQL assembled from fixed fragments with bound parameters. The LLM never writes SQL or computes a number, and any LLM wording containing a number not in the result table is discarded. Only the question, recent turns and the category catalogue reach the LLM, never customer rows.

**Anomaly detection.** Four detectors (own history, peer segment, bureau-vs-internal divergence, per-peer IsolationForest) with reasons. The label is never used; flagged accounts nonetheless default 1.59x more often.

## Known limitations
* Synthetic data; `target_default`'s horizon is not defined, and exposure is original `loan_amount` (no current balance).
* The review log and Ask audit log live in `data/state/` (ephemeral on free hosting; production would use a database).
* Offline rule-based mode understands common phrasings only; the LLM mode is broader but its output is always validated.

## Layout
```
app/app.py            dashboard            src/ingestion.py   load, DQ register, join
src/features.py       feature engineering  src/analytics.py   segments, cross-cuts, insights
src/model.py          CV, models, drivers  src/anomaly.py     detectors, review store
src/ask_portfolio.py  Ask the Portfolio    src/llm.py         OpenAI / Azure OpenAI client
src/assets.py         build + load         src/make_docs.py   regenerate the two PDFs
data/raw/             the 5 source CSVs    tests/             pytest suite
```

## Ask the Portfolio retrieval modules

The dashboard keeps `src/ask_portfolio.py` as its backwards-compatible assistant entry point.
The retrieval stages are also separated under `src/agent/` so they can be tested and extended:

- `intent.py`: deterministic empty / length / prompt-injection / small-talk gate.
- `discovery.py`: reports the loaded source names, row counts, grains and actual columns.
- `planner.py`: asks the LLM for a JSON query plan only (never SQL or Python).
- `validator.py`: validates the plan against the existing metric and dimension allow-lists.
- `executor.py`: executes allow-listed DuckDB SQL with bound filter values.
- `evidence.py`: packages result rows with source and query lineage.
- `response.py`: formats answers from calculated results and checks numeric wording.
- `orchestrator.py`: composes those stages for modular testing and future integration.

`src/ask_portfolio.py` remains the Streamlit-facing implementation so existing app behavior and
conversation state remain compatible. Query plans now support `is_null` and `not_null` dimensions;
these compile to SQL `IS NULL` and `IS NOT NULL`, while categorical values remain bound parameters.

### LLM configuration

Set `GROQ_API_KEY` in `.env` locally or in the host's secrets manager. Optional settings are
`GROQ_MODEL`, `GROQ_BASE_URL`, `LLM_TIMEOUT_SECONDS`, and `LLM_MAX_RETRIES`. If no key is set,
the app falls back to its offline rule-based planner. Credentials are never printed or included
in the repository. Do not commit `.env` or `.streamlit/secrets.toml`.
