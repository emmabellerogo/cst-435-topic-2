"""One-shot loader: GELU test-set predictions -> Supabase ``predictions`` (for the audit).

Usage (from the repo root; run as a module so ``api`` and ``shared`` import):

    python -m db.log_test_predictions --dry-run      # every check, no writes
    python -m db.log_test_predictions                # insert, then verify the view
    python -m db.log_test_predictions --verify-only  # only re-check v_fairness_audit
    python -m db.log_test_predictions --resume       # finish an interrupted insert

Reads models/gelu/test_predictions.csv (written once by api.evaluate_final; no
model is loaded, nothing is rescored) and logs each row as a labeled held-out
prediction of the served run, so v_fairness_audit can compute FPR/FNR by sex.

Before writing anything it checks:
  - the CSV has exactly the 7,327 rows and confusion matrix in evaluation.json,
    unique ids, and labels equal to calibrated proba >= 0.5
  - runs row <run-id> (default 3) is 'gelu' and is_best
  - every adult_income_id exists, has split = 'test' and the CSV's true label,
    and the CSV covers the whole test split
  - no labeled prediction for this run exists yet (refuses duplicates; the
    uq_predictions_run_adult index would also reject them)

Each inserted row:
    request_hash      sha256 of the row's 10 model features, the same hash
                      POST /predict logs for the same inputs
    predicted_label   from the CSV (threshold 0.5)
    predicted_proba   prob_calibrated from the CSV
    served_by_run_id  <run-id>
    adult_income_id   the test row (this is what makes it count in the audit)

Writes use the SERVICE-ROLE key from .env; the key is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from api.serving import request_hash
from shared.features import CATEGORICAL_COLS, FEATURE_COLS, MISSING_CATEGORY, NUMERIC_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent
GELU_DIR = REPO_ROOT / "models" / "gelu"
PREDICTIONS_CSV = GELU_DIR / "test_predictions.csv"
EVALUATION_JSON = GELU_DIR / "evaluation.json"

RUN_ID = 3
RUN_NAME = "gelu"
EXPECTED_ROWS = 7327
THRESHOLD = 0.5
BATCH_SIZE = 1000
ID_CHUNK = 300  # ids per `in.(...)` filter, keeps request URLs short
CSV_COLUMNS = ["adult_income_id", "y_true", "prob_calibrated", "predicted_label"]


# ---------------------------------------------------------------------------
# local checks (no Supabase)
# ---------------------------------------------------------------------------
def load_csv(path: Path = PREDICTIONS_CSV, evaluation_path: Path = EVALUATION_JSON) -> pd.DataFrame:
    """Read and check test_predictions.csv against evaluation.json."""
    df = pd.read_csv(path)
    problems = []
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    evaluation = json.loads(evaluation_path.read_text())

    if len(df) != EXPECTED_ROWS or evaluation["n"] != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows; CSV has {len(df)}, "
                        f"evaluation.json says {evaluation['n']}")
    if df[CSV_COLUMNS].isna().any().any():
        problems.append("CSV has empty cells")
    if df["adult_income_id"].duplicated().any():
        problems.append(f"{int(df['adult_income_id'].duplicated().sum())} duplicate adult_income_id values")
    if not df["predicted_label"].isin([0, 1]).all() or not df["y_true"].isin([0, 1]).all():
        problems.append("labels must be 0/1")
    if not df["prob_calibrated"].between(0, 1).all():
        problems.append("prob_calibrated must be in [0, 1]")
    if not ((df["prob_calibrated"] >= THRESHOLD).astype(int) == df["predicted_label"]).all():
        problems.append("predicted_label is not prob_calibrated >= 0.5 on every row")
    if evaluation.get("threshold") != THRESHOLD:
        problems.append(f"evaluation.json threshold is {evaluation.get('threshold')}")
    if not problems:
        cm = confusion_matrix(df["y_true"], df["predicted_label"])
        if cm != evaluation["confusion_matrix"]:
            problems.append(f"CSV confusion matrix {cm} != evaluation.json {evaluation['confusion_matrix']}")
    if problems:
        raise SystemExit(f"{path} failed its checks:\n  - " + "\n  - ".join(problems))
    df["adult_income_id"] = df["adult_income_id"].astype(int)
    return df


def confusion_matrix(y_true, y_pred) -> List[List[int]]:
    """[[TN, FP], [FN, TP]] as plain ints."""
    t, p = pd.Series(y_true).astype(int), pd.Series(y_pred).astype(int)
    return [[int(((t == 0) & (p == 0)).sum()), int(((t == 0) & (p == 1)).sum())],
            [int(((t == 1) & (p == 0)).sum()), int(((t == 1) & (p == 1)).sum())]]


def model_features(row: dict) -> dict:
    """The 10 features of an adult_income row, normalized as POST /predict does."""
    out = {c: int(row[c]) for c in NUMERIC_COLS}
    for c in CATEGORICAL_COLS:
        v = row[c].strip() if isinstance(row[c], str) else row[c]
        out[c] = MISSING_CATEGORY if v in (None, "", "?") else v
    return out


def build_rows(df: pd.DataFrame, sources: Dict[int, dict], run_id: int) -> List[dict]:
    return [{
        "request_hash": request_hash(model_features(sources[int(r.adult_income_id)])),
        "predicted_label": int(r.predicted_label),
        "predicted_proba": float(r.prob_calibrated),
        "served_by_run_id": run_id,
        "adult_income_id": int(r.adult_income_id),
    } for r in df.itertuples(index=False)]


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
def get_client():
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(REPO_ROOT / ".env")
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first.")
    return create_client(url, key)


def _paged(make_query: Callable, page: int = 1000) -> List[dict]:
    """All rows of an ordered query, 1000 at a time (PostgREST's response cap)."""
    rows: List[dict] = []
    while True:
        data = make_query().range(len(rows), len(rows) + page - 1).execute().data
        rows.extend(data)
        if len(data) < page:
            return rows


def check_run(client, run_id: int) -> None:
    data = (client.table("runs").select("id,name,is_best,calibration_method")
            .eq("id", run_id).limit(1).execute().data)
    if not data:
        raise SystemExit(f"runs has no row with id {run_id}")
    run = data[0]
    if run["name"] != RUN_NAME or run["is_best"] is not True:
        raise SystemExit(f"runs id {run_id} is {run['name']!r} (is_best={run['is_best']}); "
                         f"expected the is_best {RUN_NAME!r} run")
    print(f"[run] id {run_id} = {run['name']}, is_best, calibration {run['calibration_method']}")


def fetch_sources(client, df: pd.DataFrame) -> Dict[int, dict]:
    """adult_income rows for every CSV id; check each is a test row with the CSV's label."""
    cols = ",".join(["id", "split", "income_label", *FEATURE_COLS])
    ids = df["adult_income_id"].tolist()
    sources: Dict[int, dict] = {}
    for i in range(0, len(ids), ID_CHUNK):
        for row in client.table("adult_income").select(cols).in_("id", ids[i:i + ID_CHUNK]).execute().data:
            sources[int(row["id"])] = row

    problems = []
    absent = [i for i in ids if i not in sources]
    if absent:
        problems.append(f"{len(absent)} ids not in adult_income, e.g. {absent[:5]}")
    not_test = [i for i in ids if i in sources and sources[i]["split"] != "test"]
    if not_test:
        problems.append(f"{len(not_test)} ids are not split='test', e.g. {not_test[:5]}")
    truth = dict(zip(df["adult_income_id"], df["y_true"]))
    wrong = [i for i in ids if i in sources and int(sources[i]["income_label"]) != int(truth[i])]
    if wrong:
        problems.append(f"{len(wrong)} ids have a different income_label than y_true, e.g. {wrong[:5]}")
    n_test = (client.table("adult_income").select("id", count="exact")
              .eq("split", "test").limit(1).execute().count)
    if n_test != len(ids):
        problems.append(f"adult_income has {n_test} test rows but the CSV has {len(ids)}")
    if problems:
        raise SystemExit("the CSV does not match adult_income:\n  - " + "\n  - ".join(problems))
    print(f"[check] all {len(ids)} ids are split='test' with matching labels "
          f"(the whole test split)")
    return sources


def existing_labeled_ids(client, run_id: int) -> set:
    rows = _paged(lambda: client.table("predictions").select("adult_income_id")
                  .eq("served_by_run_id", run_id).not_.is_("adult_income_id", "null")
                  .order("adult_income_id"))
    return {int(r["adult_income_id"]) for r in rows}


def insert_rows(client, rows: List[dict]) -> None:
    for i in range(0, len(rows), BATCH_SIZE):
        client.table("predictions").insert(rows[i:i + BATCH_SIZE]).execute()
        print(f"[insert] {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")


def verify_audit(client, run_id: int, evaluation_path: Path = EVALUATION_JSON) -> None:
    """Check v_fairness_audit now holds sex-group rows that add up to the test evaluation."""
    linked = existing_labeled_ids(client, run_id)
    view = (client.table("v_fairness_audit").select("run_id,group_value,n,tp,fp,tn,fn,fpr,fnr")
            .eq("run_id", run_id).order("group_value").execute().data)
    expected = json.loads(evaluation_path.read_text())["confusion_matrix"]

    problems = []
    if len(linked) != EXPECTED_ROWS:
        problems.append(f"{len(linked)} labeled predictions for run {run_id}, expected {EXPECTED_ROWS}")
    groups = sorted(r["group_value"] for r in view)
    if groups != ["Female", "Male"]:
        problems.append(f"expected groups ['Female', 'Male'], got {groups}")
    totals = {k: sum(int(r[k]) for r in view) for k in ("n", "tn", "fp", "fn", "tp")}
    if totals["n"] != EXPECTED_ROWS:
        problems.append(f"view counts {totals['n']} rows, expected {EXPECTED_ROWS}")
    if [[totals["tn"], totals["fp"]], [totals["fn"], totals["tp"]]] != expected:
        problems.append(f"view totals [[{totals['tn']}, {totals['fp']}], [{totals['fn']}, "
                        f"{totals['tp']}]] != evaluation.json {expected}")
    for r in view:
        for rate in ("fpr", "fnr"):
            if r[rate] is None or not 0 <= r[rate] <= 1:
                problems.append(f"{r['group_value']}: {rate} is {r[rate]}")

    print(f"\n[audit] v_fairness_audit, run {run_id} (rates computed in SQL):")
    print(f"  {'group':<8}{'n':>6}{'tp':>6}{'fp':>6}{'tn':>6}{'fn':>6}{'FPR':>9}{'FNR':>9}")
    for r in view:
        fpr = "n/a" if r["fpr"] is None else f"{r['fpr']:.4f}"
        fnr = "n/a" if r["fnr"] is None else f"{r['fnr']:.4f}"
        print(f"  {r['group_value']:<8}{r['n']:>6}{r['tp']:>6}{r['fp']:>6}{r['tn']:>6}{r['fn']:>6}"
              f"{fpr:>9}{fnr:>9}")
    if problems:
        raise SystemExit("v_fairness_audit check failed:\n  - " + "\n  - ".join(problems))
    print("[verified] Female and Male rows present; totals match the test confusion matrix")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", type=int, default=RUN_ID, help="served runs.id (default 3)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="run every check (local and Supabase reads); insert nothing")
    mode.add_argument("--verify-only", action="store_true",
                      help="only check v_fairness_audit for the run")
    mode.add_argument("--resume", action="store_true",
                      help="insert only the ids not yet logged (after an interrupted run)")
    args = ap.parse_args()

    df = load_csv()
    print(f"[csv] {len(df)} rows; labels = prob_calibrated >= 0.5; "
          f"confusion matrix matches evaluation.json")
    client = get_client()
    check_run(client, args.run_id)
    if args.verify_only:
        verify_audit(client, args.run_id)
        return

    sources = fetch_sources(client, df)
    existing = existing_labeled_ids(client, args.run_id)
    if existing and not args.resume:
        raise SystemExit(f"predictions already has {len(existing)} labeled rows for run "
                         f"{args.run_id}. Nothing written. If an earlier run was interrupted, "
                         "use --resume; to check the audit, use --verify-only.")
    stray = existing - set(df["adult_income_id"])
    if stray:
        raise SystemExit(f"{len(stray)} logged ids for run {args.run_id} are not in the CSV, "
                         f"e.g. {sorted(stray)[:5]}; refusing to add to them")
    todo = df[~df["adult_income_id"].isin(existing)]
    rows = build_rows(todo, sources, args.run_id)
    print(f"[plan] {len(rows)} to insert, {len(existing)} already logged")

    if args.dry_run:
        print(f"[dry-run] example row: {rows[0] if rows else None}")
        print("[dry-run] nothing written to Supabase.")
        return
    insert_rows(client, rows)
    verify_audit(client, args.run_id)


if __name__ == "__main__":
    sys.exit(main())
