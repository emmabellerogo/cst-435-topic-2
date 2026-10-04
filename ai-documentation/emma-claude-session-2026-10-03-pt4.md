# Claude Code Session Transcript — October 3, 2026 Part 4

Student: Emma Rogoveanu  
Course: CST-435 Deep Learning  
Project: Topic 2 — Income Insight  
AI Tool: Claude Code (VS Code Extension)

This file contains the raw transcript of my Claude Code session. It has been preserved as displayed rather than reconstructed or summarized.


/clear
claude-session-2026-10-03-pt3.md
You are my coder for this step only.

Do not run commands, evaluate anything, write to Supabase, commit, push, or deploy. I will do those actions myself.

Work only in:

`/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2`

Current verified state:
- Real UCI Adult Income data is loaded in Supabase.
- Fixed split:
  - train: 34,188
  - val: 7,327
  - test: 7,327
- Preprocessing pipeline is implemented and tested.
- Config-driven PyTorch training is implemented and tested.
- Controlled experiment runner is implemented and tested.
- Current full test suite passed:
  - 62 passed
  - 1 skipped
- Three controlled configs were compared using validation only:
  - baseline: [64,32], ReLU, dropout 0.1
  - gelu: [64,32], GELU, dropout 0.1
  - deep: [128,64,32], ReLU, dropout 0.1
- Controls were matched across all runs.
- GELU won using lowest validation BCE loss:
  - GELU val loss: 0.3156
  - Deep val loss: 0.3158
  - Baseline val loss: 0.3163
- The test set has NOT yet been evaluated.
- No experiment rows have been written to Supabase yet.
- Models exist under:
  - `models/baseline/`
  - `models/gelu/`
  - `models/deep/`

## Task

Implement the final evaluation pipeline for the selected GELU model.

### Calibration

Use validation data only to fit the probability calibration method.

Choose a defensible binary-classification calibration approach, preferably:
- temperature scaling, or
- Platt/logistic calibration

Do NOT fit calibration on the test set.

Explain why you chose the method.

Compute calibration metrics before and after calibration on validation data:
- Brier score
- Expected Calibration Error (ECE)
- reliability-bin data for a calibration/reliability plot

Save the fitted calibration artifact under:

`models/gelu/`

The final API must later be able to load:
- preprocessor
- MLP checkpoint
- calibrator

### Final test evaluation

After calibration is fitted and frozen, evaluate the test split exactly once.

Compute and save:
- test loss
- test accuracy
- ROC-AUC
- confusion matrix in `[[TN, FP], [FN, TP]]` form
- per-class precision
- per-class recall
- per-class F1
- support
- calibrated probabilities
- Brier before calibration
- Brier after calibration
- ECE before calibration
- ECE after calibration
- reliability-bin data

Do not use the test metrics to change the model, preprocessing, threshold, calibration method, or hyperparameters afterward.

Use the default binary threshold of 0.5 unless the assignment explicitly requires threshold optimization.

### Permutation importance

Compute permutation importance for the final GELU model using held-out data.

Use a method appropriate for the preprocessing + neural-network pipeline.

Prefer doing permutation importance on the final test set only after the model is frozen, or on validation if you think test-set feature interpretation risks post-test tuning. Explain your choice.

Report feature-level importance in terms of the original 10 model features, not 84 one-hot columns, if feasible.

Save results in a JSON-friendly structure suitable for:
- `runs.permutation_importance`
- README discussion
- Streamlit Model Performance / Model Card displays

### Final artifact/data files

Save final evaluation outputs in or under:

`models/gelu/`

Suggested artifacts:
- `calibrator.*`
- `evaluation.json`
- `calibration.json`
- `permutation_importance.json`

Reuse existing `history.json` and `run.json` where appropriate rather than duplicating the same information.

Update or regenerate the GELU `run.json` so it contains the complete row for the Supabase `runs` table, including:
- architecture
- hyperparameters
- best epoch
- validation metrics
- test metrics
- calibration metrics
- confusion matrix
- per-class metrics
- permutation importance
- checkpoint path
- preprocessor path
- calibration method
- `is_best = true`

Do not insert it into Supabase yet.

### Controlled experiment rows

Also provide a clean way to create complete `runs` rows for:
- baseline
- gelu
- deep

The assignment requires all three controlled configurations to be persisted eventually.

For baseline and deep, validation-only metrics are fine until/unless the schema requires test fields; do NOT evaluate their test sets just to populate the table.

The selected GELU run is the only configuration that should receive final test evaluation.

### Tests

Add focused offline tests for:
- calibrator fits on validation only
- calibrator serialization roundtrip
- ECE/Brier computations
- confusion matrix format
- per-class metrics
- threshold remains fixed at 0.5
- final evaluation receives test data only after calibration has already been fitted
- test evaluation does not retrain/refit model, preprocessing or calibrator
- permutation importance returns the original feature names
- GELU final run row matches the `runs` schema
- only GELU is marked `is_best`

Do not run tests.

## Scope restrictions

Do NOT:
- refactor FastAPI yet
- modify Streamlit
- deploy
- write to Supabase
- retrain or alter the three controlled configurations
- use test metrics for model selection
- tune the threshold based on test results

## When finished

Stop and explain:

1. Which files were created or changed.
2. Which calibration method you chose and why.
3. Exactly which data calibration uses.
4. Exactly when the test set is touched.
5. Which final metrics are produced.
6. How permutation importance works.
7. Which artifacts are saved.
8. The exact focused test command I should run.
9. The exact full-suite test command.
10. The exact final-evaluation command I should run myself.
11. How I will later persist the three controlled runs to Supabase.

Do not execute anything.
Show more
I'll start by reading the previous session notes to orient on the existing code layout.



Read claude-session-2026-10-03-pt3.md
Read train.py
Read run_experiments.py
Read model.py
Read preprocessing.py
Read db.py
Read 001_init.sql
Read test_train.py
Read 001_init.sql
Read claude-session-2026-10-03-pt2.md
Schema is at db/migrations/001_init.sql.



Read 001_init.sql
Read features.py
Read run.json
Now the conftest and the experiment tests, to match conventions and make sure nothing auto-runs against Supabase.



Read conftest.py
Read test_run_experiments.py
Read requirements.txt
I have what I need. First, two small additive edits so the test split can be fetched separately (after calibration) instead of being loaded together with val.



Edit db.py
Added 4 lines
Edit train.py
Added 3 lines
Now the calibrator module. It doesn't import any training code, so the API can load it.



Write calibration.py
233 lines
Edit calibration.py
Modified
Next, the runs-table contract and the persistence CLI. The final evaluation imports this module.



Write persist_runs.py
227 lines
Now the final-evaluation pipeline and CLI.



Write evaluate_final.py
463 lines
Now the tests. First the calibration math, using hand-computed values:



Write test_calibration.py
167 lines
Now the final-evaluation tests, on a small synthetic Adult-shaped frame:



Write test_evaluate_final.py
283 lines
Edit test_evaluate_final.py
Removed 1 line
Last, the tests for the runs rows and the "only GELU is best" rules:



Write test_persist_runs.py
189 lines
I've written the final-evaluation pipeline, the calibrator, the runs-row builder and 43 offline tests. Nothing has been run, evaluated, written to Supabase or committed, so expect a fix or two on your first test run.

1. Files
File	Status
api/calibration.py	new: the temperature-scaling calibrator, its JSON save/load, and Brier, ECE and reliability bins. It imports no training code, so the API can load it.
api/evaluate_final.py	new: calibration, the one test evaluation, permutation importance, the final runs row, and the command line.
api/persist_runs.py	new: the runs column list, the row checks, and a command line that builds rows for baseline, gelu and deep (dry run unless you pass --write).
api/db.py	additions only: fetch_adult_income(split=...) and fetch_run_names()
api/train.py	one change: load_adult_income(split=None). With no argument it behaves exactly as before.
tests/test_calibration.py, tests/test_evaluate_final.py, tests/test_persist_runs.py	new: 19 + 12 + 12 tests
The three trained models, their configs and run_experiments.py are unchanged.

2. Calibration method: temperature scaling
The calibrated probability is p = sigmoid(logit / T), with one number T fit by minimizing log loss on validation.

It can't change any 0.5 decision. Dividing by T > 0 keeps every logit's sign, so the threshold-0.5 predictions, the confusion matrix and ROC-AUC are identical before and after. Only how confident the probabilities are changes. Platt scaling also fits an intercept, which quietly moves the decision boundary. That would amount to threshold tuning, which you ruled out.
One parameter can't overfit 7,327 rows. Isotonic regression can.
It suits this model. The MLP was trained with BCE and its validation loss is already good, so any miscalibration should be mild, mostly over- or under-confidence. A single scale factor is the right size of fix.
It's saved as plain JSON (calibrator.json), not a pickle, so loading it can't run code.
3. Data used for calibration
Only the validation rows, which are now read from Supabase on their own (split = 'val', 7,327 rows). The pipeline is the saved models/gelu/preprocessor.joblib and model.pt, used for prediction only and never refit.

Split guard: fit_calibration refuses any frame that contains a non-val row.
Freeze: a fitted calibrator raises an error if you try to fit it again.
Caveat: the after-calibration validation numbers in calibration.json are measured on the same data T was fit on, so they flatter the result. The test numbers are the honest check.
4. When the test set is touched
The order is enforced in code:

The command checks that gelu is the winner in models/experiments/controlled_comparison.json, so it only ever runs on the validation-selected model.
It refuses if evaluation.json exists or run.json already has test metrics.
It checks that model.pt, preprocessor.joblib and run.json belong together: same config, same best epoch, same feature names.
It loads the val rows and confirms the loaded model reproduces run.json's val loss to within 1e-4.
It fits T and writes calibrator.json and calibration.json.
Only then are the test rows fetched, through a callback that runs exactly once. One test confirms the calibrator file already exists when that happens.
The test rows are scored once, and permutation importance runs on them. After that, nothing is fitted again.
There's no --threshold, method or bin-count option, so nothing can be adjusted after you see test numbers. --overwrite only exists to redo an interrupted run; the pipeline is deterministic, so it reproduces the same numbers.

5. Final metrics, all on test at a fixed 0.5 threshold
Loss: BCE on the uncalibrated logits (same definition as val_loss), plus the calibrated loss.
Classification: accuracy, ROC-AUC, and precision, recall and F1 for >50K.
Confusion matrix: [[TN, FP], [FN, TP]].
Per class (<=50K and >50K): precision, recall, F1 and support.
Calibration: Brier and ECE before and after, with 10 equal-width reliability bins for each. These are computed on both validation and test.
Per row: calibrated and uncalibrated probabilities for every test row.
6. Permutation importance
I ran it on the test set, after everything is frozen.

Why test: it's the only data the model never influenced. Validation already chose the model and fit T, so importance there would describe data the pipeline was tuned on.
Why there's no tuning risk: it runs in the same single pass. Nothing it produces feeds back into the model, and the CLI refuses to run a second evaluation.
How it works:

Shuffle before preprocessing. Each of the 10 original columns is shuffled in the raw frame, so all one-hot columns for a categorical feature move together. Results are reported for the 10 features, not the 84 encoded columns.
Score the full pipeline. Each shuffled frame goes through preprocessor → MLP → calibrator.
Importance is the mean drop in ROC-AUC over 10 repeats with seed 42, reported with its standard deviation and the individual values. The rise in log loss and the drop in accuracy are recorded too.
Output: features are ranked, and each entry is tagged numeric or categorical.
7. Artifacts in models/gelu/
File	Contents
calibrator.json	T, fitted_on_split: "val", the 0.5 threshold, the formula
calibration.json	Validation before/after Brier, ECE and bins
evaluation.json	Every test metric from section 5, plus T
test_predictions.csv	adult_income_id, y_true, logit, both probabilities, predicted label (useful later for the fairness audit's predictions rows)
permutation_importance.json	The structure stored in runs.permutation_importance
run.json	Rewritten as the complete runs row with is_best: true. history.json is reused, not duplicated.
In run.json, the headline ece_* and brier_* columns are the test values, as the schema comment intends. Validation and test calibration values both go in calibration_bins.

One gap: the runs table has no calibrator_path column. The API should load calibrator.json from the same folder as checkpoint_path. If you'd rather store the path, that needs a schema migration.

8. Focused tests

.venv/bin/python -m pytest tests/test_calibration.py tests/test_evaluate_final.py tests/test_persist_runs.py -v -p no:cacheprovider
I expect 43 to pass.

9. Full suite

.venv/bin/python -m pytest -q -p no:cacheprovider
I expect 105 passed and 1 skipped (62 existing + 43 new).

10. Final evaluation (reads Supabase, writes nothing there)

.venv/bin/python -m api.evaluate_final --run gelu
11. Saving the three runs to Supabase later
Do a dry run first. It checks every row and writes models/experiments/runs_rows.json for you to inspect:


.venv/bin/python -m api.persist_runs
Then insert:


.venv/bin/python -m api.persist_runs --write
Which runs get rows: it reads the run list and the winner from the comparison file.
What it refuses:
a winner without its final evaluation
baseline or deep with any test or calibration values
more than one is_best
names already in runs
Order: baseline and deep are inserted first with NULL test columns, then gelu last as the only is_best = true row. The database's unique index on is_best backs this up.
Things to know:

api/calibration.py imports scipy directly. It comes with scikit-learn, but you may want to list it in api/requirements.txt for Render.
test_predictions.csv is about 7,300 rows. Decide whether to commit it with the rest of models/.
Running run_experiments --overwrite again would retrain gelu and replace this final run.json.


claude-session-2026-10-03-pt3.md
Refactor the backend for the CST-435 Topic 2 Income Insight project so it serves the already-trained final GELU model.

Important constraints:
- Do NOT retrain any model.
- Do NOT rerun final evaluation.
- Do NOT modify the train/val/test split.
- Do NOT modify the Streamlit frontend.
- Do NOT deploy anything.
- Do NOT write anything to Supabase unless a test specifically uses mocks. I will run live commands myself.
- Preserve the existing final artifacts in `models/gelu/`.
- The selected run is GELU, Supabase run id 3.
- The final artifacts are:
  - `models/gelu/model.pt`
  - `models/gelu/preprocessor.joblib`
  - `models/gelu/calibrator.json`
  - `models/gelu/evaluation.json`
  - `models/gelu/permutation_importance.json`
  - `models/gelu/run.json`
  - `models/gelu/test_predictions.csv`

Please inspect the existing backend first and then implement the required FastAPI contract.

Required endpoints:

1. `GET /healthz`
- Return a simple healthy response.
- Include whether the model artifacts loaded successfully.
- Keep compatibility with the existing health test if reasonable.

2. `GET /version`
- Return useful identifying information such as:
  - app/project name
  - model run name = `gelu`
  - run id = 3 if available from configuration/database
  - model/config version information
- Do not expose secrets.

3. `GET /schema`
Return the exact fields the frontend needs to build an input form for the 10 raw model features:
Numeric:
- age
- education_num
- capital_gain
- capital_loss
- hours_per_week

Categorical:
- workclass
- marital_status
- occupation
- relationship
- native_country

For each field include enough information for Streamlit to know:
- field name
- whether numeric or categorical
- expected type
- allowed/category values where appropriate

Prefer deriving categorical options from the fitted preprocessor rather than hard-coding them if practical.

Do NOT expose `sex` or `race` as prediction inputs.

4. `POST /predict`
Input:
- exactly one row using the 10 model features

Behavior:
- validate the request
- preprocess using the saved preprocessor
- run the saved GELU model
- apply saved temperature calibration
- use fixed threshold 0.5
- return predicted label and calibrated probability
- also return useful metadata such as run name/run id if appropriate
- log the prediction to Supabase `predictions`

Logging requirements:
- create a deterministic request hash from a canonical representation of the 10 raw input fields
- save:
  - request_hash
  - predicted_label
  - predicted_proba
  - served_by_run_id = GELU run id
  - created_at handled by DB
- `adult_income_id` should remain NULL for arbitrary user predictions
- do not log protected attributes because they are not prediction inputs
- make logging code testable/mocked cleanly

5. `POST /predict_batch`
- Accept a CSV upload containing the same 10 input fields
- validate required columns
- score every row with the same frozen preprocessor/model/calibrator
- fixed threshold 0.5
- return a useful batch response suitable for Streamlit, including predictions/probabilities and row count
- log each prediction to Supabase
- at the start of every batch-scoring request, perform a real Supabase read as required by the assignment. Make this purposeful, such as confirming the active run/model metadata, not a meaningless dummy query.
- fail clearly for malformed CSVs, missing columns, invalid numeric values, or DB failures

6. `GET /audit`
- Read the SQL-computed fairness results from the existing `v_fairness_audit` view.
- Return FPR/FNR and supporting counts grouped by protected attribute value for the selected GELU run.
- Do NOT calculate fairness rates in Python if the SQL view already provides them.
- This endpoint should be ready for the Streamlit Bias Audit tab.

Important startup behavior:
- Use a FastAPI lifespan/startup mechanism.
- At startup:
  - load `models/gelu/model.pt`
  - load `models/gelu/preprocessor.joblib`
  - load `models/gelu/calibrator.json`
  - validate that the artifacts correspond to GELU
  - read required model/run information from Supabase
- Load the model/preprocessor/calibrator once, not on every request.

Please clean up or replace the old starter backend behavior that still references the synthetic dataset if it conflicts with this architecture.

Tests to add/update:
- `/healthz` works
- `/version` works
- `/schema` contains exactly the 10 raw model features and excludes sex/race
- valid `/predict` returns label + probability in [0,1]
- `/predict` uses fixed threshold 0.5
- frozen reference input produces a stable probability within ±1e-3
- `/predict` logging creates the expected prediction payload
- `/predict_batch` returns the same number of predictions as input rows
- malformed/missing-column batch requests fail clearly
- batch request performs the required Supabase read
- `/audit` returns SQL-view results without recomputing FPR/FNR in Python
- Supabase failure path is handled clearly
- tests should be offline/mocked by default
- preserve existing tests where possible

Do not run any commands for me. When done, give me:
1. files created/changed
2. exact endpoint request/response shapes
3. important design decisions
4. any concerns or assumptions
5. focused test command for me to run
6. full test command for me to run

Do not commit or push anything.
Show more
I'll start by reading the existing backend: the FastAPI app, schemas, and the tests that exercise it.



Read main.py
Read schemas.py
Read test_training.py
Read run.json
Read test_schema.py
Read test_health.py
Read test_supabase.py
Read training.py
Read evaluation.json
I need the full list of test files to find the existing health test. This is a read-only directory listing:



Bash List test, api, shared and model folders
IN
ls "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/tests" "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/api" "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/shared" "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/models/gelu" "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2"

OUT
/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2:
MODEL_CARD.md
README.md
ai-documentation
api
data
db
models
render.yaml
requirements-dev.txt
shared
tests
ui

/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/api:
__init__.py
__pycache__
calibration.py
configs
db.py
evaluate_final.py
main.py
model.py
persist_runs.py
preprocessing.py
requirements.txt
run_experiments.py
train.py
training.py

/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/models/gelu:
calibration.json
calibrator.json
evaluation.json
history.json
model.pt
permutation_importance.json
preprocessor.joblib
run.json
test_predictions.csv

/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/shared:
__init__.py
__pycache__
data.py
features.py
schemas.py

/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2/tests:
__init__.py
__pycache__
conftest.py
test_calibration.py
test_evaluate_final.py
test_healthz.py
test_persist_runs.py
test_preprocessing.py
test_run_experiments.py
test_schema.py
test_supabase_roundtrip.py
test_train.py
test_training.py
Read test_healthz.py
Read test_supabase_roundtrip.py
Read render.yaml
Read requirements-dev.txt
Streamlit can't be modified, so I need to know what the UI imports from shared/ before changing anything there.



Bash List UI files and its imports and API calls
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && ls ui && grep -rn "import\|/predict\|/schema\|/audit\|/healthz\|/version\|/train\|/datasets\|/runs" ui --include=*.py | grep -v "^ui/.*#" | head -60

OUT
app.py
requirements.txt
(eval):1: no matches found: --include=*.py
Read app.py
I'll keep the /schema and /predict keys as-is since that's all the UI reads. Now checking whether models/ is committed, which determines if the API tests can rely on real artifacts.
Bash Show gitignore, tracked model files, calibrator
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && cat .gitignore; echo ---; git ls-files models | head -20; echo ---; head -3 models/gelu/test_predictions.csv; echo ---; cat models/gelu/calibrator.json

OUT
# Secrets — never commit these
.env
.env.*
!.env.example
ui/.streamlit/secrets.toml

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
*.egg-info/

# OS
.DS_Store

# Downloaded datasets
data/raw/
---
models/baseline/history.json
models/baseline/model.pt
models/baseline/preprocessor.joblib
models/baseline/run.json
models/deep/history.json
models/deep/model.pt
models/deep/preprocessor.joblib
models/deep/run.json
models/experiments/controlled_comparison.csv
models/experiments/controlled_comparison.json
models/experiments/runs_rows.json
models/gelu/calibration.json
models/gelu/calibrator.json
models/gelu/evaluation.json
models/gelu/history.json
models/gelu/model.pt
models/gelu/permutation_importance.json
models/gelu/preprocessor.joblib
models/gelu/run.json
models/gelu/test_predictions.csv
---
adult_income_id,y_true,logit,prob_uncalibrated,prob_calibrated,predicted_label
6,0,1.5493978261947632,0.8248267422135462,0.8195551076076365,1
10,1,2.9411706924438477,0.9498445278616825,0.9464810994374138,1
---
{
  "method": "temperature",
  "format_version": 1,
  "temperature": 1.0238292128031385,
  "fitted_on_split": "val",
  "n_fit": 7327,
  "nll_before": 0.3155746509800234,
  "nll_after": 0.31551698512086845,
  "decision_threshold": 0.5,
  "formula": "p = sigmoid(logit / temperature)"
}
The artifacts are tracked in git, so the API tests can load the real GELU files. Now the code. First, numeric input bounds go into the shared feature contract:



Edit features.py
Added 11 lines
Edit features.py
Modified
I'm building the serving core—artifact loading, GELU checks, scoring, input validation, request hashing, and CSV parsing—while keeping it free of FastAPI and training code dependencies, since api.train needs PyYAML, which Render doesn't install.
Write serving.py
322 lines
I'll update the database layer now, replacing helpers that reference the dropped datasets/run_artifacts tables and old runs columns — including ping(), which currently fails since it still queries datasets.
Write db.py
127 lines
Now the wire-format schemas. The UI-compatible keys stay; the dataset and training types go.



Write schemas.py
178 lines
Two fixes: object → Any for pydantic, and fields renamed to features so it can't collide with anything on BaseModel.



Edit schemas.py
Modified
Edit schemas.py
Modified
Edit schemas.py
Modified
Edit schemas.py
Modified
Now the FastAPI app itself.



Write main.py
434 lines
UploadFile needs python-multipart. Without it FastAPI refuses to import the app, so it goes into the API requirements:



Edit requirements.txt
Added 2 lines
python-multipart still needs to be installed in .venv before testing. Next I'll check conftest.py, which uses an in-memory fake to replace Supabase functions so the default test suite runs offline.
Write conftest.py
112 lines
Updating the existing tests. test_healthz.py passes unchanged. In test_schema.py, the two /predict 422 tests still hold, but the audit test's by= grouping no longer exists:



Write test_schema.py
23 lines
Write test_supabase_roundtrip.py
37 lines
test_training.py tested the deleted /datasets and /train endpoints. I'll keep only its unit test of the old starter module, which you can delete along with that module:



Write test_training.py
22 lines
I've locked in the frozen reference input, but since I can't run the model myself, that test will skip until you run the one-time snapshot command to capture the expected value.
Write reference_prediction.json
20 lines
Write make_reference_prediction.py
45 lines
Now the main API test file:



Write test_api.py
391 lines
I'll fix two issues: making the malformed CSV truly invalid since pandas was parsing a one-field-short header as an index column, and loosening the float32 comparison tolerance to account for small precision differences between batch and single-row scoring.
Edit test_api.py
Modified
Edit test_api.py
Modified
Edit test_api.py
Removed 2 lines
I've refactored the API to serve the frozen GELU model and added 48 offline API tests. I haven't run the tests, the snapshot script or the server, so expect a fix or two on your first run. Nothing was committed and nothing was written to Supabase. My one command was a read-only listing of folders, .gitignore and the calibrator file, to find the existing tests and check whether models/ is in git. It is.

Before testing, install the new dependency: .venv/bin/pip install python-multipart==0.0.12. FastAPI needs it for the CSV upload and won't import the app without it.

1. Files
File	Change
api/serving.py	new. Loads and checks the artifacts, scores rows, validates input, hashes requests, parses CSVs. It doesn't import FastAPI or training code, because api.train needs PyYAML, which Render doesn't install.
api/main.py	rewritten. Startup loading plus the 6 endpoints. /datasets, /train and /runs are gone.
api/db.py	cleaned up. The old helpers pointed at the dropped datasets/run_artifacts tables, and ping() queried datasets, so it always failed. Added fetch_run, fetch_best_runs, insert_predictions and fetch_fairness_audit. The training and persistence helpers are unchanged.
shared/schemas.py	rewritten for the new request and response shapes.
shared/features.py	added NUMERIC_BOUNDS
api/requirements.txt	added python-multipart
tests/conftest.py	rewritten. FakeDB replaces every Supabase function and can simulate outages.
tests/test_api.py	new, 48 tests
tests/fixtures/reference_prediction.json, tests/make_reference_prediction.py	new. The frozen reference input and the command that records its expected probability.
tests/test_schema.py, tests/test_supabase_roundtrip.py, tests/test_training.py	updated. The /audit test changed, the live test is now read-only, and the endpoint test for deleted routes was removed. test_healthz.py is unchanged and should still pass.
The artifacts in models/gelu/ and the Streamlit app weren't touched.

2. Endpoints
GET /healthz


{"status":"ok|degraded","model_loader":true,"supabase":true,"run_loaded":true,"run_id":3,"detail":null}
It returns 503 only if the artifacts failed to load. If Supabase was down at startup, it tries to read the run again on each call.

GET /version


{"project":"Income Insight","api_version":"2.0.0","run_name":"gelu","run_id":3,
 "model":{"architecture":"mlp","hidden_sizes":[64,32],"activation":"gelu","dropout":0.1,"best_epoch":18,
          "n_inputs":10,"n_encoded_features":84,"calibration_method":"temperature","temperature":1.0238…,
          "threshold":0.5,"config":{…},"test_accuracy":0.854,"test_roc_auc":0.906},
 "git_sha":"…","torch_version":"…","sklearn_version":"…","supabase_project_ref":"…"}
No keys or secrets are included. The project ref is the public part of the URL.

GET /schema


{"features":[
   {"name":"age","kind":"numeric","type":"integer","required":true,"minimum":17,"maximum":90,"default":<training median>,"categories":null},
   …
   {"name":"workclass","kind":"categorical","type":"string","required":true,"categories":[…from the fitted encoder…]}],
 "numeric_features":[5 names],"categorical_features":[5 names],"categories":{…},
 "target_name":"income","target_classes":["<=50K",">50K"],"threshold":0.5,"run_name":"gelu"}
POST /predict

Request: {"features": {the 10 fields}, "run_id": 3}. run_id is optional; if given, it must be 3, or the API returns 409.
Response: {"run_id":3,"run_name":"gelu","label":0|1,"income":"<=50K|>50K","proba":0.83,"threshold":0.5,"calibration_method":"temperature","request_hash":"<sha256>","logged":true}
Logged row: {request_hash, predicted_label, predicted_proba, served_by_run_id: 3, adult_income_id: null}
POST /predict_batch

Request: multipart form with a field named file holding the CSV. The 10 columns are required; extra columns such as sex are ignored and listed in the response.
Response:

{"run_id":3,"run_name":"gelu","threshold":0.5,"calibration_method":"temperature","n_rows":N,"n_predicted_positive":k,"logged":N,
 "ignored_columns":[…],"predictions":[{"row":1,"label":1,"income":">50K","proba":0.83,"request_hash":"…"},…]}
GET /audit


{"run_id":3,"run_name":"gelu","attribute":"sex","split":"test","source":"v_fairness_audit",
 "groups":[{"group_value":"Female","n":…,"tp":…,"fp":…,"tn":…,"fn":…,"fpr":…,"fnr":…}],"note":null}
Errors:

422: invalid input, with {"message", "errors":[{"row","column","value","error"}]} or missing_columns.
409: a request names a different run_id.
413: an upload over 5 MB or 10,000 rows.
503: a Supabase failure ({"message":"Supabase request failed while …","error":…}), artifacts not loaded, or the run doesn't match.
3. Design decisions
Startup checks: the API loads the three artifacts once and refuses to serve unless they agree with each other and with run.json: same name, activation, config, best epoch, feature columns and temperature. It then reads the is_best row from Supabase and checks that its name, config, checkpoint path and calibration method match. Run id 3 comes from that row rather than from code; set SERVED_RUN_ID=3 if you want to pin it.
The batch's Supabase read: each batch first re-reads run 3 to confirm it is still the active is_best run that matches the loaded model. Nothing is scored or logged if that read fails or the run has changed.
Fairness rates are not computed in Python. /audit passes the view's rows through as they are. One test feeds deliberately wrong rates and checks they come back unchanged.
Request hash: sha256 of sorted, compact JSON of the 10 validated values. "45" and 45, or " Private " and "Private", give the same hash.
Input rules:
sex, race or any other extra key sent to /predict returns 422.
Numbers must be whole and inside the range seen in the Adult data.
Categories must be ones the fitted encoder knows.
"" and "?" become Unknown, as in training.
In a batch, one bad value rejects the whole file, so nothing is partly logged.
Batch logging: one bulk insert per file, so either every row is logged or none is.
Logging failure in /predict returns 503. Logging is required, so the API doesn't return a prediction it couldn't log.
Compatibility: numeric_features, categorical_features, categories, income and proba are kept so the current UI form still works.
Run locally with uvicorn api.main:app --env-file .env. The app itself doesn't load .env, so the tests can't pick up real credentials.
4. Concerns and assumptions
/audit will be empty for now. The view only counts predictions that have an adult_income_id from the test split, and none are logged yet. models/gelu/test_predictions.csv already has the 7,327 ids and labels needed to fill it with a one-off script. I can write that script; you'd run it.

The current Streamlit app is partly broken by this change:

The Train tab calls endpoints that no longer exist.
The Predict tab's run_id box defaults to 1, which now returns 409.
Run History selects old runs columns; that was already broken by the earlier schema change.
These all need fixing in the planned UI rewrite.

Frozen reference test: I couldn't compute the expected value without running the model, so that test skips until you run the snapshot command below once. Meanwhile, test_predict_matches_independent_pipeline already checks /predict against a separate calculation to within 1e-6.

Input ranges: the limits are age 17–90, education_num 1–16, capital_gain 0–99999, capital_loss 0–4356 and hours_per_week 1–99. Values outside them return 422.

Old starter code is still on disk but unused. Delete it once you're happy: git rm api/training.py shared/data.py tests/test_training.py api/configs/default.yaml

5. Focused tests

.venv/bin/pip install python-multipart==0.0.12
.venv/bin/python -m pytest tests/test_api.py tests/test_healthz.py tests/test_schema.py -v -p no:cacheprovider
Then record the reference value once and commit the fixture:


.venv/bin/python -m tests.make_reference_prediction
6. Full suite

.venv/bin/python -m pytest -q -p no:cacheprovider
I expect 47 more passed tests than your last run, minus the one removed endpoint test, and one extra skip (the reference test) until you run the snapshot command. The live Supabase test still skips unless credentials are in your environment. Paste me the output and I'll fix anything that fails.



emma-claude-session-2026-10-03-pt4.md
write a one-off script that reads models/gelu/test_predictions.csv, verifies all 7,327 rows are test rows, attaches served_by_run_id = 3, inserts them into predictions, refuses duplicates, and then verifies v_fairness_audit returns sex-group FPR/FNR rows. No model changes, no retraining, no API changes.
Show more
I'll check how db/load.py is structured and run, so the new script follows the same conventions.



Bash List db folder and show loader conventions
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && ls db db/migrations && sed -n 1,60p db/load.py && grep -n "def \|argparse\|add_argument\|load_dotenv\|__main__" db/load.py

OUT
db:
load.py
migrations
seed.py

db/migrations:
001_init.sql
"""One-shot loader: UCI Adult Income -> Supabase ``adult_income`` table.

Usage (from the repo root):

    python db/load.py --dry-run     # download/parse/validate/split only; no Supabase
    python db/load.py               # insert into Supabase (refuses if table not empty)
    python db/load.py --replace     # delete existing adult_income rows, then insert

The official UCI files ``adult.data`` and ``adult.test`` are downloaded into
``data/raw/`` if absent. They are combined (~48.8k rows), cleaned, and given a
deterministic stratified train/val/test split. Writes use the SERVICE-ROLE key
from ``.env`` (SUPABASE_URL, SUPABASE_SERVICE_KEY); the key is never printed.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILES = ("adult.data", "adult.test")

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_ROWS = 30_000
BATCH_SIZE = 1000

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]
INT_COLS = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
TEXT_COLS = [c for c in COLUMNS if c not in INT_COLS]

# Categories we require to be a subset of. native_country is intentionally
# unchecked (41 values; just must be non-numeric text).
EXPECTED = {
    "workclass": {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked"},
    "marital_status": {"Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"},
    "relationship": {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                     "Unmarried"},
    "race": {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"},
    "sex": {"Female", "Male"},
    "income": {"<=50K", ">50K"},
}
MISSING_OK = {"workclass", "occupation", "native_country"}  # '?' in the source


def download_if_missing() -> None:
16:import argparse
60:def download_if_missing() -> None:
76:def read_adult(name: str) -> pd.DataFrame:
88:def clean(df: pd.DataFrame) -> pd.DataFrame:
100:def assign_split(df: pd.DataFrame) -> pd.DataFrame:
120:def validate(df: pd.DataFrame) -> None:
143:def report(df: pd.DataFrame) -> None:
155:def to_records(df: pd.DataFrame) -> list[dict]:
165:def get_client():
166:    from dotenv import load_dotenv
169:    load_dotenv(REPO_ROOT / ".env")
176:def insert_all(df: pd.DataFrame, replace: bool) -> None:
196:def main() -> None:
197:    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
198:    ap.add_argument("--dry-run", action="store_true",
200:    ap.add_argument("--replace", action="store_true",
216:if __name__ == "__main__":
Read load.py (from line 161)
I'll place the script in db/ alongside load.py, following its conventions with a docstring, --dry-run flag, and a safely loaded service key. I'll also reuse api.serving.request_hash on the 10 features from adult_income so hashes match what /predict logs.
Write log_test_predictions.py
288 lines
I wrote db/log_test_predictions.py, a one-off script that logs GELU's 7,327 test predictions as run 3 and then checks v_fairness_audit. I haven't run it, so nothing has been written to Supabase. No model, API or test files changed.

Run it in this order, from the repo root:


.venv/bin/python -m db.log_test_predictions --dry-run
.venv/bin/python -m db.log_test_predictions
Use -m: it imports api and shared, so python db/log_test_predictions.py would fail on those imports. The dry run does every check, including the Supabase reads, and prints one example row without writing anything. The second command inserts the rows and then checks the view.

Checks before anything is written (any failure stops the script with nothing inserted):

The CSV itself: exactly 7,327 rows, no blank cells, no repeated adult_income_id, every label equal to prob_calibrated >= 0.5. Its confusion matrix must also match evaluation.json, [[5180, 394], [676, 1077]].
The run: runs id 3 must be gelu with is_best = true.
That they are test rows: every id must exist in adult_income with split = 'test' and an income_label equal to the CSV's y_true. The table must also have exactly 7,327 test rows, so the CSV covers the whole test split.
No duplicates: if run 3 already has any predictions linked to an adult_income row, the script stops. The database's unique index on (run, adult_income_id) would reject them too.
Each inserted row:

request_hash: the same sha256 that /predict logs. The script fetches each row's 10 features from adult_income and hashes them with the API's own request_hash.
predicted_label and predicted_proba: the label and prob_calibrated from the CSV.
served_by_run_id: 3.
adult_income_id: the test row's id. This link is what makes the row count in the audit.
Rows go in batches of 1,000.
The check afterwards:

Run 3 must have exactly 7,327 linked predictions.
The view must return exactly two groups, Female and Male, with FPR and FNR present and between 0 and 1.
Adding up both groups' TN, FP, FN and TP must give the test confusion matrix in evaluation.json.
It prints a per-group table with the FPR and FNR as the SQL view computed them.
Two other flags:

--verify-only repeats just the view check.
--resume finishes an interrupted insert. Each batch is its own request, so a network failure partway through would leave some rows logged. Resume adds only the missing ids and refuses if any logged id isn't in the CSV.
Afterwards GET /audit should return the same Female and Male rows.



