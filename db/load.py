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
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[download] {name}: already present, skipping")
            continue
        url = BASE_URL + name
        print(f"[download] {name}: fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "income-insight-loader"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
        tmp.replace(dest)  # only a complete download gets the real name


def read_adult(name: str) -> pd.DataFrame:
    # adult.test starts with a junk line ("|1x3 Cross validator"); adult.data doesn't.
    skip = 1 if name == "adult.test" else 0
    df = pd.read_csv(
        RAW_DIR / name, header=None, names=COLUMNS, skiprows=skip,
        skipinitialspace=True, na_values=["?"], keep_default_na=False,
        skip_blank_lines=True, dtype={c: "string" for c in TEXT_COLS},
    )
    df["source_file"] = name
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEXT_COLS:
        df[c] = df[c].str.strip()
    # adult.test labels look like ">50K." / "<=50K."
    df["income"] = df["income"].str.rstrip(".")
    # pandas treats "?" as NA via na_values; make sure no stray "?" survives.
    for c in TEXT_COLS:
        df.loc[df[c] == "?", c] = pd.NA
    return df


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic 70/15/15 split, stratified on income x sex.

    Stratifying on sex as well as income keeps both protected groups and both
    classes represented in the test split, which the fairness audit needs.
    """
    strata = df["income"] + "|" + df["sex"]
    idx = df.index.to_numpy()
    train_val, test = train_test_split(
        idx, test_size=TEST_FRAC, random_state=SEED, stratify=strata)
    train, val = train_test_split(
        train_val, test_size=VAL_FRAC / (1 - TEST_FRAC), random_state=SEED,
        stratify=strata.loc[train_val])
    df = df.copy()
    df["split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    return df


def validate(df: pd.DataFrame) -> None:
    problems = []
    if len(df) < MIN_ROWS:
        problems.append(f"only {len(df)} rows (< {MIN_ROWS})")
    for c in INT_COLS:
        if df[c].isna().any():
            problems.append(f"{c}: {int(df[c].isna().sum())} non-numeric/missing values")
    for c, allowed in EXPECTED.items():
        bad = set(df[c].dropna().unique()) - allowed
        if bad:
            problems.append(f"{c}: unexpected categories {sorted(bad)}")
    for c in TEXT_COLS:
        if c not in MISSING_OK and df[c].isna().any():
            problems.append(f"{c}: unexpected missing values")
    if not df["education_num"].between(1, 16).all():
        problems.append("education_num outside 1..16")
    if not df["age"].between(0, 120).all():
        problems.append("age outside 0..120")
    if problems:
        raise SystemExit("Validation FAILED:\n  - " + "\n  - ".join(problems))
    print(f"[validate] OK: {len(df)} rows, all expected categories present/valid")


def report(df: pd.DataFrame) -> None:
    print(f"\nRows total: {len(df)}")
    print("Rows per source file:\n" + df["source_file"].value_counts().to_string())
    print("\nSplit counts:\n" + df["split"].value_counts().to_string())
    print("\nTarget (income) counts:\n" + df["income"].value_counts().to_string())
    print("\nIncome by split:\n" + pd.crosstab(df["split"], df["income"]).to_string())
    print("\nSex by split:\n" + pd.crosstab(df["split"], df["sex"]).to_string())
    miss = df[COLUMNS].isna().sum()
    print("\nMissing values per column (non-zero only):\n"
          + (miss[miss > 0].to_string() if miss.any() else "none"))


def to_records(df: pd.DataFrame) -> list[dict]:
    out = df[COLUMNS + ["split"]].astype(object)
    out = out.where(out.notna(), None)  # pd.NA / NaN -> None (JSON null)
    records = out.to_dict(orient="records")
    for r in records:
        for c in INT_COLS:
            r[c] = int(r[c])
    return records


def get_client():
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv(REPO_ROOT / ".env")
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first.")
    return create_client(url, key)


def insert_all(df: pd.DataFrame, replace: bool) -> None:
    client = get_client()
    existing = client.table("adult_income").select("id", count="exact").limit(1).execute().count or 0
    if existing and not replace:
        raise SystemExit(
            f"adult_income already has {existing} rows. Nothing written. "
            "Re-run with --replace to delete them first.")
    if existing and replace:
        print(f"[replace] deleting {existing} existing rows "
              "(predictions.adult_income_id links to them become NULL)")
        client.table("adult_income").delete().gt("id", 0).execute()

    records = to_records(df)
    for i in range(0, len(records), BATCH_SIZE):
        client.table("adult_income").insert(records[i:i + BATCH_SIZE]).execute()
        print(f"[insert] {min(i + BATCH_SIZE, len(records))}/{len(records)}")
    final = client.table("adult_income").select("id", count="exact").limit(1).execute().count
    print(f"[done] adult_income now holds {final} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="download, parse, validate and report; do not touch Supabase")
    ap.add_argument("--replace", action="store_true",
                    help="delete existing adult_income rows before inserting")
    args = ap.parse_args()

    download_if_missing()
    df = clean(pd.concat([read_adult(n) for n in FILES], ignore_index=True))
    validate(df)
    df = assign_split(df)
    report(df)

    if args.dry_run:
        print("\n[dry-run] nothing written to Supabase.")
        return
    insert_all(df, args.replace)


if __name__ == "__main__":
    sys.exit(main())
