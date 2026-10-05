# Income Insight

**Will this person earn more than \$50K a year?** Income Insight answers that for
one person or a whole spreadsheet in seconds, using a neural network trained on the
real UCI *Adult Income* census data (48,842 people). Each answer includes a
probability estimate evaluated for calibration on held-out data. The app shows how the
network works, how it was evaluated, and where it makes different kinds of mistakes
for women and men. It is a CST-435 Topic 2 teaching project, not a tool for real
hiring, lending or eligibility decisions.

## Live deployment

| Tier | Platform | URL |
|------|----------|-----|
| **Source** | GitHub | <https://github.com/emmabellerogo/cst-435-topic-2> |
| **UI** | Streamlit Community Cloud | <https://cst-435-topic-2-oxbelz7s9ehhkfyfxkmpki.streamlit.app/> |
| **API** | Render (FastAPI) | <https://income-insight-api-0n96.onrender.com> (try [`/healthz`](https://income-insight-api-0n96.onrender.com/healthz), [`/version`](https://income-insight-api-0n96.onrender.com/version), [`/docs`](https://income-insight-api-0n96.onrender.com/docs)) |
| **Data** | Supabase Postgres | project ref `avdohphpqletgsoevrwm` (`https://avdohphpqletgsoevrwm.supabase.co`) |

The Render free tier sleeps when idle, so the first request can take about a minute.

---

## Architecture

```
┌────────────────────────────┐  HTTPS / JSON   ┌──────────────────────────────┐  service-role key  ┌──────────────────────────────┐
│ Streamlit Community Cloud  │ ──────────────► │ FastAPI on Render            │ ─────────────────► │ Supabase Postgres            │
│ ui/app.py (6 tabs)         │                 │ api/main.py                  │  reads + writes    │                              │
│ thin client: no torch,     │                 │ frozen GELU MLP (run 3)      │                    │ adult_income  (48,842 rows)  │
│ no sklearn, no model code  │                 │ + sklearn preprocessor       │                    │ runs          (3 runs)       │
│                            │                 │ + temperature calibrator     │                    │ predictions   (audit log)    │
│                            │                 │ loaded once from models/gelu │                    │ v_fairness_audit (SQL view)  │
│                            │                 └──────────────────────────────┘                    │                              │
│                            │ ── anon key, SELECT only: runs + v_fairness_audit ────────────────► │ RLS on every table           │
└────────────────────────────┘                                                                     └──────────────────────────────┘
```

- **Streamlit (UI)** never loads the model. Every prediction is an HTTPS call to the API.
  With the public **anon** key it may only `SELECT` from `runs` and from the aggregate
  `v_fairness_audit` view. Without that key it falls back to the committed files in `models/`.
- **FastAPI (Render)** serves one frozen model. It never trains. At startup it loads
  `models/gelu/` and refuses to serve unless those files agree with the `is_best` row in
  `runs`. It is the only component that writes to Supabase, using the service-role key.
- **Supabase** is the system of record for the dataset, the runs and the prediction log.
  FPR and FNR are computed in SQL by `v_fairness_audit`.

## The six tabs

| Tab | What it shows |
|-----|---------------|
| **Concepts** | Forward propagation and matrix shapes of the served 84 → 64 → 32 → 1 network. Training (q = σ(z), BCE) is kept separate from inference (p = σ(z/T)). Backpropagation derived step by step with the chain rule and the shape of every error signal and gradient. XOR: why one linear boundary cannot work, a hand-built network, one full numerical backpropagation step (loss, δs, gradients, update) on a separate 2-2-1 ReLU teaching network, an interactive XOR trainer, and ReLU vs GELU. |
| **Score a Row** | A form built from the API's `/schema`. Returns the class, the calibrated probability and the logging status. |
| **Score CSV** | Upload up to 5 MB / 10,000 rows to `/predict_batch` and download the scored file. Any invalid value rejects the whole file. |
| **Model Performance** | Test metrics, a train/validation/test comparison, learning curves, confusion matrix, per-class metrics, calibration, the three-configuration comparison and permutation importance. |
| **Bias Audit** | FPR/FNR by sex from `/audit` (SQL), the same view read directly with the anon key, an interpretation, and proposed mitigations (none implemented). |
| **Model Card** | A one-page summary built from `/version`, `/schema`, `/audit` and the stored results (separate from, but consistent with, [MODEL_CARD.md](MODEL_CARD.md)). |

## Data and preprocessing

- **Dataset:** UCI *Adult* (Kohavi & Becker, 1994 U.S. Census extract). `db/load.py`
  downloads `adult.data` and `adult.test` and combines them into **48,842 rows**. It
  cleans the data (`?` → NULL, trailing `.` removed from test labels) and loads the
  rows into `adult_income`.
- **Split:** one fixed 70 / 15 / 15 split (seed 42), **stratified on income × sex**,
  stored in the `split` column. The splits are **train 34,188 · validation 7,327 ·
  test 7,327**, and every experiment uses this same split.
- **Model inputs (10):** numeric `age`, `education_num`, `capital_gain`,
  `capital_loss`, `hours_per_week`; categorical `workclass`, `marital_status`,
  `occupation`, `relationship`, `native_country`.
- **Not model inputs:** `sex` and `race` are protected attributes, kept only for the
  audit. `education` duplicates `education_num`, and `fnlwgt` is a census sampling weight.
- **Pipeline** (`api/preprocessing.py`), fitted on **train rows only** (the code refuses any other split):
  - numeric columns: median imputation, then `StandardScaler`
  - categorical columns: missing values become `"Unknown"`, then `OneHotEncoder(handle_unknown="ignore")`
  - result: **84 encoded features**

## Model, training and selection

- **Architecture:** a PyTorch MLP, `[Linear → activation → Dropout] × N → Linear(1)`,
  trained with `BCEWithLogitsLoss` and AdamW.
- **Checkpoint choice:** after each epoch the model is scored on validation, and the
  epoch with the **lowest validation loss** is kept. Early stopping uses patience 5.
- **Selection rule (fixed in advance):** lowest validation BCE; ties go to the higher
  validation ROC-AUC, then the config name. The test split played no part in
  training, early stopping, checkpoint choice or model selection.

### Three controlled configurations (validation set)

Controlled across all three: same split and preprocessor, seed 42, learning rate 1e-3,
weight decay 1e-4, batch size 256, 30-epoch budget, early-stopping patience 5,
dropout 0.1. Each comparison against `baseline` changes exactly one thing.

| Rank | Config | Hidden sizes | Activation | Best epoch | Val loss | Val acc | Val F1 (>50K) | Val ROC-AUC |
|------|--------|--------------|------------|-----------:|---------:|--------:|--------------:|------------:|
| 1 ✅ | `gelu` | [64, 32] | GELU | 18 | **0.3156** | 0.8534 | 0.6658 | 0.9070 |
| 2 | `deep` | [128, 64, 32] | ReLU | 9 | 0.3158 | 0.8557 | 0.6639 | 0.9074 |
| 3 | `baseline` | [64, 32] | ReLU | 18 | 0.3163 | 0.8567 | 0.6688 | 0.9074 |

**Provenance of this table:** the table above was produced by running
[`db/queries/runs_comparison.sql`](db/queries/runs_comparison.sql) (the query below) as
a read-only `SELECT` in the Supabase SQL Editor against the live `runs` table on
2026-10-04. Emma ran it and pasted the result rows; its ranks and every metric match this
table. The same query output also lists the controlled variables (seed 42, 30-epoch
budget, patience 5, batch 256, learning rate 0.001, weight decay 0.0001, dropout 0.1 for
all three runs). The runs were first trained and persisted by `api/run_experiments.py` and
`api/persist_runs.py`, and the Python comparison in
`models/experiments/controlled_comparison.json` agrees with the SQL result. The SQL
output itself was not saved as a file in this repository.

**Interpreting the result.** `gelu` won on the pre-registered criterion, but the
spread in validation loss is only 0.0007 (0.3156 vs 0.3163), from a single seed. The
three configurations are effectively tied. `baseline` has slightly higher validation
accuracy and F1, and both ReLU runs have slightly higher ROC-AUC (0.9074 vs 0.9070). GELU's smooth, non-zero
gradient for negative inputs is a plausible reason for its marginally lower loss, but
one seed cannot separate that from run-to-run noise. The evidence supports "GELU is at
least as good here, chosen by a rule fixed in advance", not "GELU is better". Only
`gelu` was ever scored on test; the other runs' test columns are deliberately NULL.

### SQL query for the runs comparison

[`db/queries/runs_comparison.sql`](db/queries/runs_comparison.sql) generates the
comparison table from the `runs` table. It is a single read-only `SELECT`, and it also
returns the controlled variables (seed, epoch budget, patience, batch size, learning
rate, weight decay) so the output itself shows the controls were matched. The query
was checked as follows:

- Every column it uses was checked against `db/migrations/001_init.sql`.
- It parses as one `SelectStmt` with the PostgreSQL parser (`pglast`/libpg_query).
- Its ordering, simulated over `models/experiments/runs_rows.json`, gave the ranking
  above, and the live execution then returned the same ranking.

```sql
select
    row_number() over (
        order by r.val_loss asc, r.val_roc_auc desc, r.name asc
    )                                                                   as rank,
    r.id,
    r.name,
    r.hidden_sizes,
    r.activation,
    r.dropout,
    r.seed,
    r.epochs                                                            as epoch_budget,
    (r.config ->> 'early_stopping_patience')::int                       as patience,
    r.batch_size,
    r.learning_rate,
    r.weight_decay,
    r.best_epoch,
    round(r.val_loss::numeric, 4)                                       as val_loss,
    round(r.val_accuracy::numeric, 4)                                   as val_accuracy,
    round(((r.val_metrics ->> 'f1')::double precision)::numeric, 4)     as val_f1,
    round(r.val_roc_auc::numeric, 4)                                    as val_roc_auc,
    round(r.test_accuracy::numeric, 4)                                  as test_accuracy,
    round(r.test_roc_auc::numeric, 4)                                   as test_roc_auc,
    r.is_best
from runs r
where r.name in ('baseline', 'gelu', 'deep')
order by rank;
```

## Selected model: `gelu` (Supabase run 3)

| | |
|---|---|
| Network | 84 → 64 → 32 → 1, GELU, dropout 0.1, 7,553 parameters, best epoch 18 |
| Calibration | temperature scaling, **T = 1.0238**, fitted on the **validation** split only; served probability p = σ(z / T) |
| Decision rule | `>50K` when p ≥ 0.5 (fixed, never tuned) |

**Test set (7,327 rows, evaluated once after selection)**

| Accuracy | ROC-AUC | Precision (>50K) | Recall (>50K) | F1 (>50K) | BCE (uncal. → cal.) |
|---------:|--------:|-----------------:|--------------:|----------:|--------------------:|
| 0.8540 | 0.9060 | 0.7322 | 0.6144 | 0.6681 | 0.3190 → 0.3189 |

- **Confusion matrix** `[[TN, FP], [FN, TP]] = [[5180, 394], [676, 1077]]`. Per class,
  `<=50K` has precision 0.8846, recall 0.9293, F1 0.9064 (support 5,574), and `>50K`
  has precision 0.7322, recall 0.6144, F1 0.6681 (support 1,753).
- **Which class is harder:** `>50K`. Only 1,753 of 7,327 test rows (23.9%) are `>50K`.
  The model finds 1,077 of them (recall 61.4%), against 5,180 of the 5,574 `<=50K`
  rows (recall 92.9%). Its 676 false negatives outnumber its 394 false positives, so
  most mistakes fall on the minority class. The training loss weights every row
  equally, and there are about three `<=50K` rows per `>50K` row. That is consistent
  with the lower recall, but it does not prove the imbalance causes it, since some people
  in both classes have near-identical recorded features. The 0.5 threshold was fixed,
  not tuned. A lower threshold would trade more false positives for higher `>50K` recall.
- **Train / validation / test** (same checkpoint, uncalibrated, threshold 0.5): the
  accuracies are 0.8614 / 0.8534 / 0.8540 and the BCE values 0.2983 / 0.3156 / 0.3190.
  That is a small generalization gap.
- **Calibration on test:** ECE 0.0100 → 0.0099 and Brier 0.1014 → 0.1014. The network
  was already well calibrated, so T stayed close to 1.
- **Permutation importance** (test split, 10 repeats, mean drop in ROC-AUC):
  1. `marital_status` 0.0549
  2. `capital_gain` 0.0370
  3. `age` 0.0335
  4. `education_num` 0.0327
  5. `occupation` 0.0175
  6. `hours_per_week` 0.0138
  7. `relationship` 0.0108
  8. `capital_loss` 0.0051
  9. `workclass` 0.0039
  10. `native_country` 0.0016
- **Reference profile:** a 45-year-old married, `Exec-managerial` husband in the
  private sector working 45 hours, with education 13, no capital gain or loss, from
  the United States. The model predicts **`>50K`, p ≈ 0.8153**. The value is frozen in
  `tests/fixtures/reference_prediction.json` and checked by the tests.

Sources: `models/gelu/run.json`, `evaluation.json`, `calibrator.json`,
`permutation_importance.json`, `history.json`.

## Fairness audit (SQL)

`v_fairness_audit` joins `predictions` to `adult_income` and counts only predictions
linked to a **labeled test row** (`adult_income_id` set, `split = 'test'`). It
computes FPR = FP/(FP+TN) and FNR = FN/(FN+TP) per sex. All 7,327 test predictions of
run 3 were logged for this purpose by `db/log_test_predictions.py`. Predictions made
in the app have no true label and never enter these rates.

The table below was read from the live `GET /audit` on 2026-10-04.

| Group | Test rows | Actually >50K | TP | FP | TN | FN | FPR | FNR |
|-------|----------:|--------------:|---:|---:|---:|---:|----:|----:|
| Female | 2,429 | 265 (10.9%) | 155 | 58 | 2,106 | 110 | 0.0268 | **0.4151** |
| Male | 4,898 | 1,488 (30.4%) | 922 | 336 | 3,074 | 566 | **0.0985** | 0.3804 |

Women who earn >50K are missed more often (FNR 41.5% vs 38.0%). Men who earn ≤50K are
wrongly flagged >50K more often (FPR 9.9% vs 2.7%). The model never sees `sex`, but
`relationship` and `marital_status` act as proxies, and the base rates differ
substantially. Possible mitigations are listed in [MODEL_CARD.md](MODEL_CARD.md);
**none has been implemented**.

## What is and is not persisted

**Persisted**

| Where | What |
|-------|------|
| Supabase `adult_income` | All 48,842 UCI rows, including `sex`/`race` (audit only) and the fixed `split` |
| Supabase `runs` | One row per configuration (`deep`, `baseline`, `gelu`): hyperparameters, full config, train/val metrics. **Only `gelu` (id 3, `is_best`)** also has test metrics, calibration data, confusion matrix, per-class metrics, permutation importance and artifact paths |
| Supabase `predictions` | One row per scored request from `/predict` and `/predict_batch`: `request_hash` (SHA-256 of the 10 feature values), `predicted_label`, `predicted_proba`, `served_by_run_id`, `created_at`. `adult_income_id` is NULL for app predictions and set only for the 7,327 labeled test predictions used by the audit |
| Git, `models/<run>/` | `model.pt`, `preprocessor.joblib`, `history.json` (per-epoch train/val curves), `run.json`; for `gelu` also `calibrator.json`, `calibration.json`, `evaluation.json`, `permutation_importance.json`, `test_predictions.csv` |
| Git, `models/experiments/` | `controlled_comparison.json/.csv`, `runs_rows.json` |

**Not persisted**

- Raw feature values of app predictions (only their hash). The hash is not
  anonymization, because the input space is small enough to guess.
- Uploaded CSV files and scored CSVs. These are returned to the browser only.
- Any user identity, account or session. The app has no login.
- Model weights and per-epoch histories in Supabase. These are in Git only, and the
  API loads the weights from `models/gelu/`.
- Anything from training: the API cannot train, and its retired `/train` and
  `/datasets` endpoints no longer exist.

## Setup (local)

Requires Python 3.11 (Render uses 3.11.9).

```bash
git clone https://github.com/emmabellerogo/cst-435-topic-2.git
cd cst-435-topic-2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt          # both tiers + pytest

cp .env.example .env                         # SUPABASE_URL + SUPABASE_SERVICE_KEY (API / scripts only)
cp ui/.streamlit/secrets.toml.example ui/.streamlit/secrets.toml   # API_URL (+ optional anon key)
```

Never put the service-role key in the UI secrets; the UI uses only the anon key.

**Run the two tiers** (the API writes prediction logs to the Supabase in `.env`):

```bash
uvicorn api.main:app --reload --port 8000 --env-file .env                 # terminal 1
API_URL=http://localhost:8000 streamlit run ui/app.py                     # terminal 2, from the repo root
```

Optional API variables: `MODEL_DIR` (default `models/gelu`), `SERVED_RUN_ID` (pin a
`runs.id`), `ALLOWED_ORIGINS` (CORS, default `*`). Optional UI variables:
`SUPABASE_URL` and `SUPABASE_ANON_KEY`.

## Reproducing the pipeline

Each step uses the service-role key from `.env`. The committed artifacts already
contain the results, so **do not re-run these against the live project** unless you
intend to replace them.

```bash
# 1. Database: paste db/migrations/001_init.sql into the Supabase SQL Editor and run it.
python db/load.py --dry-run          # download + validate + split, no writes
python db/load.py                    # load adult_income (refuses if not empty)

# 2. Train the three controlled configurations (train + validation only)
python -m api.run_experiments api/configs/baseline.yaml api/configs/gelu.yaml api/configs/deep.yaml
#    single run:  python -m api.train --config api/configs/gelu.yaml

# 3. Calibrate on validation, then evaluate the winner on test exactly once
python -m api.evaluate_final --run gelu

# 4. Persist the three runs rows (dry run first)
python -m api.persist_runs
python -m api.persist_runs --write

# 5. Log the labeled test predictions so v_fairness_audit has data
python -m db.log_test_predictions --dry-run
python -m db.log_test_predictions
```

## Tests

```bash
pytest -q
```

The suite runs offline. Supabase is replaced by an in-memory fake, the API tests load
the real `models/gelu/` artifacts, and the UI tests use Streamlit's `AppTest` with a
fake API. The suite covers:

- preprocessing (train-only fitting)
- training and the experiment runner
- calibration and the final evaluation
- run persistence
- the API contract, validation and logging
- the frozen reference prediction
- the UI tabs, including the numerical XOR backpropagation step, which is checked
  against finite differences

### Live Supabase tests (`tests/test_supabase_roundtrip.py`)

Both tests skip unless `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set, so CI never
touches the live project. Prediction-logging behavior is tested offline (mocked) in
`tests/test_api.py`. Those tests show that the API *calls* the insert with the right
row, but not that a row reaches Supabase.

| Test | Live effect | Status |
|------|-------------|--------|
| `test_live_startup_reads_gelu_run_and_audit_view` | read-only: `/healthz`, `/version`, `/audit` | **Passed** locally against the live project on 2026-10-04 |
| `test_live_predict_writes_a_predictions_row` | **writes one row**: `POST /predict` with the frozen reference profile, then reads `predictions` back and checks the label, probability, run id and `adult_income_id IS NULL` | Needs a second opt-in, `RUN_LIVE_WRITE_TESTS=1`. **Passed** when Emma ran it against the live project on 2026-10-04 (`2 passed`, output pasted by Emma; one `predictions` row was written) |

```bash
set -a; source .env; set +a                                   # service-role key, never committed
python -m pytest -rs tests/test_supabase_roundtrip.py         # read-only test only
RUN_LIVE_WRITE_TESTS=1 python -m pytest -rs tests/test_supabase_roundtrip.py   # adds the one-row write test
```

The written row has `adult_income_id` NULL, so it can never enter the fairness audit.
It is left in place.

### Continuous integration (GitHub Actions)

[`.github/workflows/tests.yml`](.github/workflows/tests.yml) runs the same offline
suite on every push to any branch and on every pull request to `main`. It can also be
started by hand from the Actions tab.

- **Environment:** Python 3.11 on Ubuntu.
- **Install:** the CPU-only build of the `torch` version pinned in
  `api/requirements.txt`, then `pip install -r requirements-dev.txt`.
- **Run:** `python -m pytest -q -rs`.
- **No secrets:** the workflow has none and only reads the repository. It never
  contacts the live Supabase project or the Render API, so the live round-trip test
  is skipped, and `-rs` prints the skip reason in the log.

Recorded runs (read with `gh run list` on 2026-10-04):
- [run 37240404601](https://github.com/emmabellerogo/cst-435-topic-2/actions/runs/37240404601):
  `db4bbf6`, success
- [run 37241871026](https://github.com/emmabellerogo/cst-435-topic-2/actions/runs/37241871026):
  `2891d46`, success
- [run 37245649136](https://github.com/emmabellerogo/cst-435-topic-2/actions/runs/37245649136):
  `8bee7fa`, success, `209 passed, 1 skipped`

Run 37245630017 (`93e6c0a`) was cancelled by the concurrency rule when the next push
arrived. It did not fail. Before the trigger change above, pushes to other branches
(such as `komal-frontend`) were not tested.

## Deployment

1. **Supabase:** run `db/migrations/001_init.sql`, then steps 1, 4 and 5 above. This
   enables RLS, revokes `anon`/`authenticated` access to the tables, and grants `anon`
   `SELECT` on `runs` and `v_fairness_audit` only.
2. **Render:** New → Blueprint → this repo (`render.yaml`). Set `SUPABASE_URL` and
   `SUPABASE_SERVICE_KEY` as secret env vars. The health check is `/healthz`.
3. **Streamlit Community Cloud:** main file `ui/app.py`. In Secrets set `API_URL` (the
   Render URL), plus optionally `SUPABASE_URL` and `SUPABASE_ANON_KEY` (anon key only).
   `.streamlit/config.toml` at the repo root sets the 5 MB upload limit.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Artifacts loaded, Supabase reachable, served run confirmed |
| `GET` | `/version` | Served run, architecture, temperature, library versions, build SHA |
| `GET` | `/schema` | The 10 input fields, ranges, training medians and allowed categories |
| `POST` | `/predict` | One row → calibrated P(>50K), label, logged to `predictions` |
| `POST` | `/predict_batch` | CSV upload (≤ 5 MB, ≤ 10,000 rows) → one logged prediction per row |
| `GET` | `/audit` | FPR/FNR by sex for the served run, passed through from `v_fairness_audit` |

## Project structure

```
api/        FastAPI service (main.py, serving.py, db.py) and the offline pipeline
            (preprocessing, model, train, run_experiments, calibration, evaluate_final, persist_runs)
api/configs baseline.yaml · gelu.yaml · deep.yaml
db/         migrations/001_init.sql · queries/runs_comparison.sql · load.py · log_test_predictions.py
shared/     features.py (feature contract) · schemas.py (API models)
ui/         app.py + one module per tab, api_client.py, perf_data.py
models/     committed artifacts for the three runs + experiments/
tests/      pytest suite (offline)
.github/workflows/tests.yml   CI: runs the offline suite on every push and on PRs to main
ai-documentation/   AI-use transcripts
```

Leftovers from the course template are still in the repo but are **not used** by the
current app: `api/training.py`, `shared/data.py` (synthetic generator), `db/seed.py`
and `api/configs/default.yaml`.

---

## Team

Both members built their parts with AI assistance (Claude / Claude Code). Each person
directed the work, made the decisions, and reviewed and tested the results, while
much of the code and documentation text was generated by the AI tools. The session
transcripts are in [`ai-documentation/`](ai-documentation/).

**Evidence key:** Ⓖ means supported by the Git history in this repository (commit
shown). Ⓔ means confirmed by Emma but not verifiable from the repository alone (it
happened on GitHub, Render, Streamlit Cloud or a local machine).

### Emma Rogoveanu

- **Individual report:** [reports/emma-report.md](reports/emma-report.md)
- **Video:** <https://www.loom.com/share/2e1f5555063d4be2b7c983668f2c66b0>
- **Contributions:**
  - Set up the team repository and the backend and data tier with Claude Code:
    - the Adult Income schema and Supabase loader
    - train-only preprocessing and config-driven PyTorch training
    - the three-configuration controlled comparison
    - calibration, the one-time test evaluation, permutation importance and run persistence
    - the frozen GELU FastAPI service, including its deployment on Render
    - prediction logging and the labeled SQL fairness audit
  - After the initial frontend was in place, used Claude Code to make and verify the
    final rubric fixes in the UI:
    - learning curves and the train/validation/test table
    - the direct anon-key Supabase audit display
    - corrections to the Concepts tab
    - the Score CSV upload limit and template
    - the experiment-chart fix

    Also used Claude Code to rewrite this README and the model card, and to add the
    GitHub Actions workflow. Reviewed the results, merged them into `main`, and checked
    the public deployment.
- **Task list:**
  - [x] Set up the team repository and development environment. Ⓖ `2cd7778`
  - [x] Implement the Adult Income database schema and load the dataset into Supabase. Ⓖ `49cba46`
  - [x] Implement training-only preprocessing and configurable PyTorch MLP training. Ⓖ `5fc0197`, `cf7fb96`
  - [x] Run and compare three controlled model configurations. Ⓖ `72efe69`
  - [x] Complete calibration, final evaluation, feature importance, and run persistence. Ⓖ `cc42073`
  - [x] Implement and deploy the FastAPI model-serving backend. Ⓖ `378cb7f`; deployment Ⓔ
  - [x] Implement prediction logging and the labeled SQL fairness audit. Ⓖ `49cba46`, `378cb7f`, `1c974b1`
  - [x] Coordinate and review the final UI rubric fixes using Claude Code: learning curves, split metrics, direct Supabase audit display, Concepts corrections, and CSV improvements. Ⓖ `b2ec7b4`
  - [x] Update technical README and model-card documentation with Claude Code. Ⓖ `b2ec7b4`
  - [x] Configure the local UI's anon-key Supabase connection and verify the direct audit matches `/audit`. Ⓔ
  - [x] Merge the final changes into main and verify the public deployment's updated performance and audit sections. Ⓖ merge `32d542f`; public check Ⓔ
  - [x] Verify the public template CSV scores correctly and results download. Ⓔ
  - [x] Verify the public app opens without signing in. Ⓔ
  - [x] Add the GitHub Actions test workflow with Claude Code and confirm its GitHub run passes. Ⓖ `db4bbf6`; passing run Ⓔ
  - [x] Finish and link Emma's individual report. [`reports/emma-report.md`](reports/emma-report.md)
  - [x] Record and link Emma's individual demonstration video. [Loom](https://www.loom.com/share/2e1f5555063d4be2b7c983668f2c66b0)
  - [x] Complete and verify Emma's AI-use records and contribution evidence.

### Komal Khan

- **Individual report:** [reports/komal-report.md](reports/komal-report.md)
- **Video:** <https://www.loom.com/share/bfb13bc55b424131b43442f91dbbbba5>
- **Contributions:**
  - Built the initial six-tab Streamlit frontend in a Claude session, against Emma's
    API handoff. Komal's commit `f34683e` adds `ui/`, and
    [`komal-streamlit-session-01.md`](ai-documentation/komal-streamlit-session-01.md)
    shows the files being generated in that session:
    - the API client and settings helpers
    - all six tab modules: Concepts with the XOR demo, Score a Row, Score CSV, Model
      Performance, Bias Audit and Model Card
  - Added the initial frontend test suite in the same commit (`tests/ui_fakes.py`,
    `tests/test_ui_api_client.py`, `tests/test_ui_data.py`, `tests/test_ui_app.py`).
  - The later UI fixes, documentation rewrite and CI workflow are Emma's (see above),
    not Komal's, even though one of those commits (`b2ec7b4`) was pushed to the
    `komal-frontend` branch.
- **Task list:**
  - [x] Build the initial six-tab Streamlit frontend. Ⓖ `f34683e` (authored by Komal), session transcript
  - [x] Implement the initial Concepts, Score a Row, Score CSV, Model Performance, Bias Audit, and Model Card interfaces. Ⓖ `f34683e`, session transcript
  - [x] Add frontend tests. Ⓖ `f34683e`. The tests were written in the session; the transcript does not record them being run, so Komal should confirm
  - [x] Finish and link Komal's individual report.
  - [x] Record and link Komal's individual demonstration video.
  - [x] Complete and verify Komal's AI-use records and contribution evidence.

## AI use

Transcripts of the AI-assisted sessions are in [`ai-documentation/`](ai-documentation/).
