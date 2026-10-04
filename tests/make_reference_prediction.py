"""Write the frozen reference probability into tests/fixtures/reference_prediction.json.

Run ONCE, from the repo root, after the GELU artifacts are final:

    python -m tests.make_reference_prediction

It scores the fixture's input with models/gelu/ (no Supabase, no training) and
stores the calibrated probability. tests/test_api.py then fails if /predict
ever drifts more than ``tolerance`` from it. Refuses to overwrite an existing
value unless --overwrite is given, so the snapshot stays frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from api import serving

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "reference_prediction.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overwrite", action="store_true", help="replace an existing snapshot")
    args = ap.parse_args()

    ref = json.loads(FIXTURE.read_text())
    if ref.get("expected_proba") is not None and not args.overwrite:
        raise SystemExit(f"{FIXTURE} already holds expected_proba={ref['expected_proba']}; "
                         "pass --overwrite only if the served model intentionally changed")
    art = serving.load_artifacts()
    checked = serving.validate_records([ref["features"]], art.categories)
    if checked.errors:
        raise SystemExit(f"reference input is invalid: {checked.errors}")
    proba = float(serving.predict_proba(art, checked.records)[0])
    ref.update(run_name=art.run_name, expected_proba=proba,
               expected_label=int(serving.predict_labels([proba])[0]))
    FIXTURE.write_text(json.dumps(ref, indent=2) + "\n")
    print(f"[reference] {art.run_name}: P(>50K) = {proba:.6f}, label {ref['expected_label']} -> {FIXTURE}")


if __name__ == "__main__":
    main()
