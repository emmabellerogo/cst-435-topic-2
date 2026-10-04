# Claude Code Session Transcript — October 3, 2026

Student: Emma Rogoveanu  
Course: CST-435 Deep Learning  
Project: Topic 2 — Income Insight  
AI Tool: Claude Code (VS Code Extension)

This file contains the raw transcript of my Claude Code session. It has been preserved as displayed rather than reconstructed or summarized.
You are helping me build my CST-435 Topic 2 group project, “Income Insight.”

My working repository is:

`/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2`

The professor’s original reference template is:

`/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/deep-learning-resources/Deep Learning/Project Templates/Topic_2_income-insight`

DO NOT modify the professor’s reference template. Only modify my working repository.

I am Emma. My partner is Komal. I am doing the majority of the backend/model/database work. Komal will mainly handle the Streamlit frontend.

## Important workflow rules

Before changing files:
1. Inspect the existing repository.
2. Explain briefly what you plan to change.
3. Make only changes needed for the current task.
4. Do not commit or push anything unless I explicitly tell you to.
5. Do not deploy anything unless I explicitly tell you to.
6. Do not invent test results. Only document tests that were actually run.
7. Do not fabricate AI-use documentation or pretend actions happened when they did not.
8. Preserve useful parts of the professor’s starter code when possible instead of rebuilding unnecessarily.
9. Keep security-sensitive values in `.env`. Never place Supabase service-role credentials in source code or documentation.

## Current environment

Python 3.11 is installed and the project has a `.venv`.

Use:

`.venv/bin/python`

The current baseline tests previously passed with:

`.venv/bin/python -m pytest -q -p no:cacheprovider`

Result before project modifications:

`8 passed, 1 skipped`

The `.env` file exists and contains the new Topic 2 Supabase project credentials.

Supabase project reference:

`avdohphpqletgsoevrwm`

Do not expose its service-role key.

## Current task

Replace the starter synthetic database design with the real schema required by the Income Insight assignment.

The existing `db/migrations/001_init.sql` is based on synthetic data and currently creates:

- `datasets`
- `runs`
- `run_artifacts`
- `predictions`

Do NOT use the synthetic `datasets` approach.

We need a schema appropriate for the real UCI Adult Income dataset.

### Required database design

Create an `adult_income` table with one row per Adult dataset example.

It must support the actual Adult Income fields, including the target and protected attribute needed for auditing. Use appropriate PostgreSQL types.

The Adult dataset fields should account for:

- age
- workclass
- fnlwgt
- education
- education_num
- marital_status
- occupation
- relationship
- race
- sex
- capital_gain
- capital_loss
- hours_per_week
- native_country
- income

Include a stable primary key such as `id`.

Design a `runs` table that can support multiple controlled neural-network experiments.

It must preserve enough information for:

- model architecture
- hidden-layer sizes
- activation
- dropout
- learning rate
- weight decay
- epochs
- batch size
- random seed
- train/validation/test metrics
- calibration-related metrics where appropriate
- confusion-matrix/per-class metrics where appropriate
- identifying the selected/best run
- created timestamp

Use JSONB where it makes the schema cleaner for architecture, hyperparameters, or detailed metric dictionaries, but keep important searchable fields explicit when helpful.

Create a `predictions` table that satisfies the assignment requirement:

- request_hash
- predicted_label
- predicted_proba
- served_by_run_id
- created_at

Also design it so we can correctly perform a reproducible fairness audit.

Important fairness constraint:

Arbitrary user predictions do not have verified true income labels. We therefore cannot calculate FPR/FNR from random user-entered predictions alone.

We need a way for predictions made against a labeled held-out Adult dataset row to preserve a link back to that source row so SQL can compare:

- true income label
- predicted label
- protected attribute such as sex

A nullable foreign key such as `adult_income_id` is acceptable if that is the best design.

Do not pretend a request hash gives us the true label.

Create appropriate indexes and foreign-key relationships.

Apply Row Level Security thoughtfully:

- FastAPI uses the service-role key server-side.
- Streamlit may later use the Supabase anon key for specifically approved read-only views.
- Do not expose raw privileged database access unnecessarily.
- Do not make sensitive server-side tables anonymously writable.

Do not apply the migration to Supabase yet unless I explicitly tell you to.

For this step, modify only the SQL migration and any absolutely necessary supporting schema files. Do not begin model training or frontend work yet.

After making the change:
1. Show me exactly what files changed.
2. Explain the schema in simple terms.
3. Point out any design decisions I should understand.
4. Tell me what the next logical step is.
5. Do not move on to that next step until I approve it.

## Full project direction for future tasks

Eventually this project must include:

- real UCI Adult Income dataset with at least 30,000 rows
- one-shot `db/load.py`
- sklearn Pipeline with:
  - categorical encoding
  - numeric scaling
  - missing-value imputation
- fit preprocessing only on training data
- serialized preprocessing pipeline
- PyTorch MLP with at least TWO hidden layers
- configurable:
  - hidden sizes
  - activation
  - dropout
  - learning rate
  - weight decay
  - epochs
- configs in `api/configs/`
- CLI training script `api/train.py`
- best checkpoint in `models/`
- run metrics persisted to Supabase
- train/validation/test split
- probability calibration without using the final test set for calibration
- confusion matrix
- per-class precision/recall/F1
- calibration plot/data
- permutation importance
- at least three controlled configurations using matched seeds/splits/budgets
- FastAPI endpoints:
  - POST `/predict`
  - POST `/predict_batch` accepting CSV
  - GET `/schema`
  - GET `/audit`
  - GET `/healthz`
  - GET `/version`
- model and preprocessing loaded at API startup
- prediction hashing/logging
- SQL-based FPR/FNR audit by sex or another protected attribute
- regression, schema, batch, Supabase-write, and Supabase-failure tests
- CI
- Render deployment
- Streamlit deployment
- Supabase deployment

Komal will mainly handle the six Streamlit tabs, so keep backend response formats clean and easy for another person to consume.

## AI documentation requirement

Keep a factual running record during this coding session of:

- my requests/instructions
- files you inspected
- files you created or modified
- major code or architecture decisions you suggested
- code you generated
- commands I asked you to execute, if any
- commands actually executed
- tests actually executed
- actual results of those tests
- errors encountered
- fixes made
- tasks that remain unfinished

Do not claim anything was tested, deployed, committed, pushed, or verified unless it actually happened.

At the END of our coding work, when I explicitly tell you the session is finished, create:

`ai-documentation/`

and create a Markdown file inside it named something like:

`claude-session-YYYY-MM-DD.md`

The file should contain:

- Date
- Project
- AI tool used: Claude
- Student: Emma Rogoveanu
- Summary of what AI was used for
- My substantive prompts/instructions
- Files changed
- AI-generated or AI-assisted code/work
- Technical decisions made with AI assistance
- Commands executed
- Tests executed and their actual results
- Errors/fixes
- What I personally reviewed or decided
- Remaining work
- Clear statement that the documentation reflects only actions actually performed during the session

Do not fabricate a verbatim transcript if one is unavailable. A faithful structured activity record is preferable to invented dialogue.

Do not create the AI documentation file until I explicitly say we are done with the coding session.

Bash
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && git status --short && find . -path ./.venv -prune -o -path ./.git -prune -o -type f -print | head -80 && grep -rniE "datasets|run_artifacts|001_init|hidden_dim|roc_auc" --include=*.py --include=*.md --include=*.yml --include=*.yaml --include=*.toml --include=*.sql . --exclude-dir=.venv --exclude-dir=.git -l

OUT
./render.yaml
./.DS_Store
./MODEL_CARD.md
./README.md
./requirements-dev.txt
./.gitignore
./.env
./.gitattributes
./.env.example
./ui/requirements.txt
./ui/app.py
./tests/conftest.py
./tests/__init__.py
./tests/test_schema.py
./tests/test_supabase_roundtrip.py
./tests/test_training.py
./tests/test_healthz.py
./shared/__init__.py
./shared/schemas.py
./shared/data.py
./db/seed.py
./api/db.py
./api/requirements.txt
./api/__init__.py
./api/main.py
./api/training.py
./ui/.streamlit/secrets.toml.example
./api/configs/default.yaml
./db/migrations/001_init.sql
(eval):1: no matches found: --include=*.py
Bash
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && grep -rnE "datasets|run_artifacts|predictions|001_init" api db tests shared README.md render.yaml | head -50

OUT
db/seed.py:21:MIGRATION = Path(__file__).parent / "migrations" / "001_init.sql"
db/seed.py:26:    # 001_init.sql once in the Supabase SQL Editor (see the main TUTORIAL). This
api/main.py:5:  * read datasets from Supabase before training,
api/main.py:93:# datasets
api/main.py:95:@app.post("/datasets", response_model=Dataset, tags=["datasets"])
api/main.py:189:    return PredictBatchResponse(run_id=req.run_id, predictions=items)
api/main.py:209:    """Group a run's logged predictions by a categorical feature and report the
api/main.py:215:    rows = db.predictions_for_run(run_id)
api/db.py:8:The fitted model artifact is kept in a SEPARATE ``run_artifacts`` table (no anon
api/db.py:43:    """Return True if the Supabase client can reach the datasets table."""
api/db.py:45:        get_client().table("datasets").select("id").limit(1).execute()
api/db.py:52:# datasets
api/db.py:64:        .table("datasets")
api/db.py:83:        .table("datasets")
api/db.py:93:# runs (metrics)  +  run_artifacts (model blob)
api/db.py:124:    client.table("run_artifacts").insert(
api/db.py:145:        .table("run_artifacts")
api/db.py:167:# predictions (the audit log)
api/db.py:172:        .table("predictions")
api/db.py:181:def predictions_for_run(run_id: int) -> List[dict]:
api/db.py:184:        .table("predictions")
db/migrations/001_init.sql:1:-- 001_init.sql
db/migrations/001_init.sql:6:-- datasets: one row per synthetic tabular dataset (records + labels as JSONB).
db/migrations/001_init.sql:9:create table if not exists datasets (
db/migrations/001_init.sql:26:    dataset_id  bigint      not null references datasets (id) on delete cascade,
db/migrations/001_init.sql:40:-- run_artifacts: the fitted (preprocessor + MLP) blob, base64-encoded.
db/migrations/001_init.sql:43:create table if not exists run_artifacts (
db/migrations/001_init.sql:50:-- predictions: one row per served /predict call (the prediction / audit log).
db/migrations/001_init.sql:52:create table if not exists predictions (
db/migrations/001_init.sql:63:create index if not exists idx_predictions_run_id on predictions (run_id);
db/migrations/001_init.sql:69:-- bypasses RLS entirely, so writes (and reads of artifacts/predictions/datasets)
db/migrations/001_init.sql:72:alter table datasets      enable row level security;
db/migrations/001_init.sql:74:alter table run_artifacts enable row level security;
db/migrations/001_init.sql:75:alter table predictions   enable row level security;
tests/test_supabase_roundtrip.py:27:            "/datasets", json={"name": "roundtrip", "n_rows": 500, "noise": 1.0}
tests/conftest.py:22:        "datasets": {},
tests/conftest.py:25:        "predictions": [],
tests/conftest.py:27:    counters = {"datasets": 0, "runs": 0}
tests/conftest.py:30:        counters["datasets"] += 1
tests/conftest.py:32:            "id": counters["datasets"],
tests/conftest.py:41:        store["datasets"][row["id"]] = row
tests/conftest.py:45:        return store["datasets"].get(dataset_id)
tests/conftest.py:78:            "id": len(store["predictions"]) + 1,
tests/conftest.py:84:        store["predictions"].append(row)
tests/conftest.py:87:    def predictions_for_run(run_id):
tests/conftest.py:88:        return [p for p in store["predictions"] if p["run_id"] == run_id]
tests/conftest.py:98:    monkeypatch.setattr(db, "predictions_for_run", predictions_for_run)
README.md:30:and the API exposes a fairness `/audit` view over logged predictions.
README.md:45:| Tables | datasets · runs · predictions | datasets · runs · **run_artifacts** · predictions |
README.md:57:│  thin client, no ML  │                 │  MLP + sklearn pipeline   │                  │  datasets/runs/  │
I'll rewrite db/migrations/001_init.sql directly since the old tables were never applied to your new Supabase project, making that cleaner than adding a 002 migration. I'll leave api/db.py, db/seed.py, and the tests untouched since they still reference the old schema.
Write 001_init.sql
175 lines
I rewrote db/migrations/001_init.sql and changed nothing else. I haven't run it against Supabase, run any tests, or committed anything. I also don't know what "I authenticated" referred to. I didn't need any authentication for this step, so tell me if you meant something specific.

The schema in simple terms
adult_income has one row per Adult example. It has all 15 fields plus these extras:
income_label, a 0/1 value computed from income, so SQL never has to compare strings.
split, one of train, val or test.
The ? markers become NULL in workclass, occupation and native_country, so the sklearn imputer sees real missing values.
runs has one row per experiment.
The hyperparameters are plain columns: hidden sizes as an integer array (at least two layers is enforced), activation, dropout, learning rate, weight decay, epochs, batch size and seed.
The headline val/test metrics are plain columns too, so you can sort by them.
Calibration has ECE and Brier before and after, plus the reliability-bin data.
The confusion matrix, per-class metrics, permutation importance and the full config are JSONB.
is_best marks the selected run, and a unique index allows only one at a time.
predictions has request_hash, predicted_label, predicted_proba, served_by_run_id and created_at. It also has a nullable adult_income_id linking back to the source row.
v_fairness_audit is a view that computes TP, FP, TN, FN, FPR and FNR by sex for each run.
Design decisions to understand
Ground truth comes only from adult_income_id. The request hash is for traceability and can't give us a label. Predictions from the live /predict endpoint leave that link NULL, so they're excluded from FPR/FNR. The audit only counts predictions your code makes against held-out rows.
split is fixed on adult_income. db/load.py will assign it once with a fixed seed. Every config then trains on identical rows, which matches your "matched splits" requirement, and the audit joins only split = 'test' rows. The catch is that you can't change the split per run without changing this design.
One prediction per run per source row. A unique index enforces this, so re-running the audit can upsert instead of piling up duplicates.
run_artifacts is gone. The model and preprocessor go to files in models/, and runs stores the paths. If you'd rather keep the model blob in Supabase for Render, we can add that table back.
Access control.
RLS is on for all three tables, and I revoked anon and authenticated privileges on them.
Anon gets read-only access to runs (metrics only) and to the aggregate audit view, which has no individual rows.
FastAPI's service-role key bypasses RLS, so adult_income and predictions stay server-side.
Old code now breaks. api/db.py, db/seed.py, api/main.py, tests/conftest.py and the README table still assume datasets, dataset_id and run_artifacts. I left them alone because you scoped this step to SQL only.
Next logical step
Review the SQL and tell me if you want any changes. For example, drop the anon read of the audit view, or add a run_artifacts table. Once you approve it, I'd write db/load.py: download the real Adult data, check it has at least 30,000 rows, assign the fixed split and insert the rows. I'll wait for your go-ahead before doing that or applying the migration.



001_init.sql#1-175
Before I run this, tell me exactly how `db/load.py` gets the UCI Adult Income dataset.

I want to know:
- the exact source it uses
- whether it downloads the files automatically or expects me to download them
- the filenames it expects
- whether it uses the official UCI train/test files or a Python package
- what command I should run
- what files should appear locally, if any

Do not change any code yet. Just explain the current loader behavior so I understand it.
Show more
Bash
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && ls db; ls data 2>&1 | head; grep -rniE "adult|uci|openml|fetch_" --include=*.py --include=*.md --include=*.txt . --exclude-dir=.venv --exclude-dir=.git | head; git status --short

OUT
migrations
seed.py
ls: data: No such file or directory
(eval):1: no matches found: --include=*.py
 M db/migrations/001_init.sql
db/load.py doesn't exist yet, so there is no loader behavior to explain. I haven't written it, and I haven't changed any code since the migration.

I checked the repo to confirm:

db/ holds only migrations/ and the professor's old seed.py. seed.py generates synthetic data and has nothing to do with Adult.
There is no data/ folder.
Nothing in the Python or Markdown files mentions Adult, UCI, OpenML or fetch_.
The only modified file is db/migrations/001_init.sql.
Your questions are design choices I haven't made yet. Here is what I'd propose, for you to approve or change:

Question	Proposal
Source	The official UCI files, adult.data (32,561 rows) and adult.test (16,281 rows), from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/. Together that's about 48,842 rows, above your 30,000 minimum.
Download	Automatic. If the files are missing locally, the loader downloads them with urllib or requests. If you'd rather download them yourself, it can skip the download when the files are already there.
Filenames	adult.data and adult.test, saved under data/raw/. I'd add data/raw/ to .gitignore.
Parsing	pandas with the 15 column names from the field list. It would convert ? to NULL, strip whitespace, skip the junk first line of adult.test, and remove the trailing . from its income labels (>50K.).
Python package	None. The OpenML or fetch_openml route is not used, so everything comes from the official files.
Split	It assigns split (train/val/test) with a fixed seed, stratified on income, across all the rows combined. It would not reuse UCI's own train/test boundary, because we need a validation set as well.
Command	.venv/bin/python db/load.py (maybe with --dry-run to parse and validate without writing to Supabase).
Local files afterward	data/raw/adult.data and data/raw/adult.test.
Credentials	It reads SUPABASE_URL and the service-role key from .env and inserts in batches.
One thing to settle first: the loader inserts into adult_income, so the migration has to be applied to Supabase before you run it. That means you'd need to approve applying it first.

Do you want me to write db/load.py as proposed? If so, tell me whether to download automatically or have you place the files yourself, and I'll build it with a --dry-run mode and test it locally before anything touches Supabase.



Yes, write `db/load.py` using the proposed design.

Use the official UCI Adult Income files:
- `adult.data`
- `adult.test`

Download them automatically from the official UCI source if they are not already present, and save them under:

`data/raw/`

Also:
- add `data/raw/` to `.gitignore`
- include a `--dry-run` mode so I can validate the dataset locally without writing anything to Supabase
- do not run the loader for me
- do not apply the migration
- do not upload anything
- do not commit or push anything

The loader should:
- combine the official training and test files
- handle the junk first line in `adult.test`
- strip trailing periods from `adult.test` income labels
- convert `?` to missing values / `None`
- assign a deterministic stratified train/val/test split with a fixed seed
- validate at least 30,000 rows and the expected categories
- insert to Supabase in batches only when I run it without `--dry-run`
- refuse to duplicate existing rows unless I use an explicit `--replace` option
- print row counts, split counts, target counts, and missing-value counts

After writing it, explain the code to me and stop. Do not execute anything.
Show more
Bash
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && cat .gitignore | head -30; cat .env.example; sed -n 1,60p db/seed.py; cat requirements-dev.txt api/requirements.txt; sed -n 1,45p api/db.py

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
# Copy to .env for local development of the API tier (never commit .env).
# The API uses the SERVICE-ROLE key (server-side, full access).
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key

# Optional: comma-separated list of allowed CORS origins for the API.
# Default "*" is fine for a teaching demo.
ALLOWED_ORIGINS=*
"""Apply the migration (manually) and seed one default dataset into Supabase.

Usage (from the repo root, with a .env present or the env vars exported):

    python -m db.seed

This uses the SERVICE-ROLE key so it can write. Never commit that key.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from api import db
from shared.data import FEATURE_COLS, generate_tabular

load_dotenv()

MIGRATION = Path(__file__).parent / "migrations" / "001_init.sql"


def main() -> None:
    # NOTE: supabase-py cannot run arbitrary DDL over the REST API. Apply
    # 001_init.sql once in the Supabase SQL Editor (see the main TUTORIAL). This
    # script only seeds data, which the REST API supports.
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first.")

    print(f"Migration file: {MIGRATION} (apply it in the SQL Editor if you have not).")

    n_rows = 2000
    records, labels = generate_tabular(n_rows=n_rows, noise=1.0, seed=42)
    positive_rate = sum(labels) / len(labels)
    row = db.insert_dataset(
        name="default-adult",
        n_rows=n_rows,
        n_features=len(FEATURE_COLS),
        positive_rate=positive_rate,
        records=records,
        labels=labels,
    )
    print(
        f"Seeded dataset id={row['id']} name={row['name']} "
        f"n_rows={n_rows} positive_rate={positive_rate:.3f}"
    )


if __name__ == "__main__":
    main()
# Local development + running the test suite from the repo root.
# Installs both tiers plus pytest. On the actual clouds, each tier installs only
# its own requirements (api/requirements.txt or ui/requirements.txt).
-r api/requirements.txt
-r ui/requirements.txt
pytest==8.3.3
httpx==0.27.2
PyYAML==6.0.2
# API tier (deployed on Render.com)
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
# CPU-only torch keeps the Render build small and free-tier friendly.
torch==2.4.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
supabase==2.7.4
python-dotenv==1.0.1
"""Supabase persistence helpers for the API tier.

All Supabase access for the model service is funneled through this module. The
FastAPI app calls these functions; it never talks to Supabase directly. The
Streamlit UI NEVER imports this module -- it does its one read-only query with
the anon key on its own side.

The fitted model artifact is kept in a SEPARATE ``run_artifacts`` table (no anon
RLS policy) so the large base64 blob is never exposed to the public anon key --
only run metrics are anon-readable.

Environment variables (set locally in a .env, and in the Render dashboard):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> the service-role key (server-side only, secret!)
"""
from __future__ import annotations

import os
from typing import List, Optional

from supabase import Client, create_client

_client: Optional[Client] = None

# Columns returned to callers as a "run" (metrics only, no model blob).
_RUN_COLS = (
    "id,dataset_id,hidden_dim,lr,batch_size,epochs,"
    "accuracy,precision,recall,f1,roc_auc,created_at"
)


def get_client() -> Client:
    """Lazily create and cache a Supabase client."""
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
    return _client


def ping() -> bool:
    """Return True if the Supabase client can reach the datasets table."""
    try:
        get_client().table("datasets").select("id").limit(1).execute()
Write load.py
218 lines
Bash
IN
cd "/Users/emmarogoveanu/Desktop/school/search engines & data mines/projects/cst-435-topic-2" && printf '\n# Downloaded datasets\ndata/raw/\n' >> .gitignore && tail -5 .gitignore

OUT
# OS
.DS_Store

# Downloaded datasets
data/raw/
I wrote db/load.py and added data/raw/ to .gitignore. I haven't run or tested anything, applied the migration, uploaded anything, or committed. The code has never been executed, so expect to fix a bug or two on the first dry run.

How the loader works
Download. download_if_missing() fetches adult.data and adult.test from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/ into data/raw/. It skips any file that's already there. Each download goes to a .part file first and is renamed only when complete, so an interrupted download can't leave a half-file under the real name.

Parse. read_adult() uses pandas with the 15 column names and skipinitialspace=True.

It treats ? as missing and skips blank lines.
It skips the first line of adult.test.
It tags each row with its source_file, which is for reporting only and isn't inserted.
Clean. clean() strips whitespace and removes the trailing . from income labels (>50K. becomes >50K).

Validate. validate() stops with a list of problems if any of these fail:

fewer than 30,000 rows
non-numeric values in the integer columns
unexpected categories in workclass, marital_status, relationship, race, sex or income
missing values in columns that shouldn't have any (only workclass, occupation and native_country may)
education_num outside 1–16
age outside 0–120
native_country isn't checked against a list.

Split. assign_split() makes a 70/15/15 train/val/test split with random_state=42. It's stratified on income and sex combined, not income alone, so both sexes and both classes appear in the test split. That matters for the fairness audit. If you want income-only stratification, it's a one-line change.

Report. It prints total rows, rows per source file, split counts, income counts, income-by-split and sex-by-split tables, and missing values per column.

Insert. This runs only without --dry-run.

get_client() loads .env and uses the service-role key, which is never printed.
If adult_income already has rows, it exits with nothing written unless you pass --replace.
--replace deletes the existing rows first. The on delete set null setting then clears the adult_income_id link on any predictions pointing at them.
It inserts in batches of 1000, with missing values sent as JSON null.
Things to know
The dry run needs no .env and no Supabase, because that code is imported only when needed. It does need internet on the first run, and it needs pandas and scikit-learn, which are already in the venv requirements.
The old UCI ml/machine-learning-databases/ path may redirect. urllib follows redirects, but if the download fails, paste me the error.
The Adult data has about 50 exact duplicate rows. I kept them, because they're legitimate records with their own ids.
ids come from the database in insertion order. Rows go in as adult.data first, then adult.test.
To run it yourself
From the repo root:


.venv/bin/python db/load.py --dry-run
Check that the row count is about 48,842, the split is about 70/15/15, and the target is about 24% >50K. Inserting needs the migration applied first, then .venv/bin/python db/load.py.

I'm stopping here. Tell me what the dry run prints, or tell me when you want to apply the migration.