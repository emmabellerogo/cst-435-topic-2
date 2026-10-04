"""Controlled comparison of several training configs, judged on validation only.

Usage (from the repo root, with a .env holding the service-role key):

    python -m api.run_experiments api/configs/baseline.yaml api/configs/gelu.yaml \\
        api/configs/deep.yaml --reuse-existing

Steps:
  1. load every config; refuse duplicate names (they would share models/<name>/)
  2. refuse configs whose controlled hyperparameters differ (CONTROLLED_FIELDS)
  3. decide per config: train, reuse an existing run (--reuse-existing) or
     retrain over it (--overwrite); anything else that exists is an error
  4. read adult_income once, fit the preprocessor on train rows once, and train
     each config with the existing api.train pipeline (train + val only)
  5. rank the runs by SELECTION_CRITERION and print one comparison table
  6. save the table to <output-dir>/experiments/<summary-name>.json and .csv

This script never evaluates the test split and never writes to Supabase: it has
no --evaluate-test or --write-run option, and test rows are not passed to
training. Final test evaluation of the winner is a separate, later step.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from api.train import (
    MODELS_DIR,
    TrainConfig,
    _repo_relative,
    build_run_row,
    load_adult_income,
    load_config,
    prepare_splits,
    save_artifacts,
    train_model,
)

# Must be identical across every config so differences are attributable to the
# varied fields. The split and the preprocessor are shared automatically because
# the data is loaded and preprocessed once for all runs.
CONTROLLED_FIELDS = (
    "architecture", "epochs", "batch_size", "seed",
    "learning_rate", "weight_decay", "early_stopping_patience",
)
VARIED_FIELDS = ("hidden_sizes", "activation", "dropout")

SELECTION_CRITERION = (
    "lowest validation loss (BCE at each run's best-val-loss epoch); "
    "ties broken by higher validation ROC-AUC, then config name"
)
SUMMARY_DIRNAME = "experiments"  # <output-dir>/experiments/<summary-name>.{json,csv}
MIN_CONFIGS = 2


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------
def load_experiment_configs(paths: Sequence) -> List[TrainConfig]:
    """Load every config; reject duplicate names, which would share an output folder."""
    if len(paths) < MIN_CONFIGS:
        raise ValueError(f"a comparison needs at least {MIN_CONFIGS} configs, got {len(paths)}")
    configs = [load_config(p) for p in paths]
    first_path: Dict[str, str] = {}
    problems = []
    for path, cfg in zip(paths, configs):
        if cfg.name == SUMMARY_DIRNAME:
            problems.append(f"{path}: name {cfg.name!r} is reserved for the summary folder")
        if cfg.name in first_path:
            problems.append(f"name {cfg.name!r} is used by both {first_path[cfg.name]} and {path}")
        first_path.setdefault(cfg.name, str(path))
    if problems:
        raise ValueError("configs would overwrite each other's models/<name>/ outputs:\n  - "
                         + "\n  - ".join(problems))
    return configs


def check_controls(configs: Sequence[TrainConfig]) -> Dict[str, object]:
    """Raise unless every CONTROLLED_FIELDS value is identical; return the shared values."""
    problems = []
    for field in CONTROLLED_FIELDS:
        values = {cfg.name: getattr(cfg, field) for cfg in configs}
        if len(set(values.values())) > 1:
            problems.append(f"{field}: {values}")
    if problems:
        raise ValueError("controlled hyperparameters differ between configs:\n  - "
                         + "\n  - ".join(problems))
    return {field: getattr(configs[0], field) for field in CONTROLLED_FIELDS}


def plan_runs(configs: Sequence[TrainConfig], output_dir: Path,
              overwrite: bool = False, reuse_existing: bool = False) -> Dict[str, str]:
    """Return {config name: 'train' | 'reuse'}. Never silently replaces a saved run.

    - no models/<name>/model.pt         -> 'train'
    - exists, overwrite=True            -> 'train' (replaces it)
    - exists, reuse_existing=True       -> 'reuse', only if its run.json was
                                           trained from exactly this config
    - exists, neither flag              -> error
    """
    if overwrite and reuse_existing:
        raise ValueError("choose either overwrite or reuse_existing, not both")
    plan: Dict[str, str] = {}
    problems = []
    for cfg in configs:
        out_dir = Path(output_dir) / cfg.name
        if not (out_dir / "model.pt").exists() or overwrite:
            plan[cfg.name] = "train"
        elif reuse_existing:
            run_path = out_dir / "run.json"
            if not run_path.exists():
                problems.append(f"{run_path} is missing, so {cfg.name} cannot be reused")
            elif json.loads(run_path.read_text()).get("config") != asdict(cfg):
                problems.append(f"{run_path} was trained with a different config than "
                                f"{cfg.name}'s YAML; retrain it with --overwrite")
            else:
                plan[cfg.name] = "reuse"
        else:
            problems.append(f"{out_dir / 'model.pt'} already exists. Pass --reuse-existing "
                            "to use it unchanged or --overwrite to retrain it.")
    if problems:
        raise ValueError("\n  - ".join(["cannot plan runs:", *problems]))
    return plan


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def run_experiments(
    configs: Sequence[TrainConfig],
    plan: Dict[str, str],
    output_dir: Path,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    preprocessor: Optional[Pipeline] = None,
    log: Callable[[str], None] = print,
) -> List[dict]:
    """Train (or reuse) each config and return its runs-table row.

    Takes no test data at all; every returned row has NULL test columns unless
    it was reused from an earlier run.json.
    """
    rows = []
    for cfg in configs:
        out_dir = Path(output_dir) / cfg.name
        if plan[cfg.name] == "reuse":
            log(f"[{cfg.name}] reusing {_repo_relative(out_dir)}/run.json (not retrained)")
            rows.append(json.loads((out_dir / "run.json").read_text()))
            continue
        log(f"[{cfg.name}] training: hidden_sizes={cfg.hidden_sizes} "
            f"activation={cfg.activation} dropout={cfg.dropout}")
        result = train_model(cfg, X_train, y_train, X_val, y_val, log=log)
        ckpt_path, pre_path = save_artifacts(out_dir, cfg, result, preprocessor)
        row = build_run_row(cfg, result, ckpt_path, pre_path)  # no test metrics
        (out_dir / "run.json").write_text(json.dumps(row, indent=2, allow_nan=False))
        log(f"[{cfg.name}] best epoch {result.best_epoch}, val_loss {result.best_val_metrics['loss']:.4f}"
            f" -> {_repo_relative(out_dir)}/")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize_run(row: dict) -> dict:
    """The comparison-table fields of one runs-table row (validation metrics only)."""
    val = row["val_metrics"]
    return {
        "name": row["name"],
        "hidden_sizes": list(row["hidden_sizes"]),
        "activation": row["activation"],
        "dropout": row["dropout"],
        "best_epoch": row["best_epoch"],
        "val_loss": val["loss"],
        "val_accuracy": val["accuracy"],
        "val_precision": val["precision"],
        "val_recall": val["recall"],
        "val_f1": val["f1"],
        "val_roc_auc": val["roc_auc"],
    }


def _missing(v: Optional[float]) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _rank_key(s: dict):
    loss, auc = s["val_loss"], s["val_roc_auc"]
    return (
        _missing(loss), 0.0 if _missing(loss) else loss,   # lowest val loss first; missing last
        math.inf if _missing(auc) else -auc,                # then highest val AUC
        s["name"],
    )


def rank_runs(summaries: Sequence[dict]) -> List[dict]:
    """Sort by SELECTION_CRITERION and add a 1-based ``rank``; rank 1 is the winner."""
    ranked = sorted(summaries, key=_rank_key)
    if not ranked or _missing(ranked[0]["val_loss"]):
        raise ValueError("no run has a finite validation loss; cannot pick a winner")
    return [{"rank": i, **s} for i, s in enumerate(ranked, start=1)]


def _fmt4(v: Optional[float]) -> str:
    return "n/a" if _missing(v) else f"{v:.4f}"


SUMMARY_COLUMNS = [  # (header, key, formatter)
    ("rank", "rank", str),
    ("config", "name", str),
    ("hidden_sizes", "hidden_sizes", lambda v: "[" + ", ".join(str(h) for h in v) + "]"),
    ("activation", "activation", str),
    ("dropout", "dropout", lambda v: f"{v:g}"),
    ("best_epoch", "best_epoch", str),
    ("val_loss", "val_loss", _fmt4),
    ("val_acc", "val_accuracy", _fmt4),
    ("val_prec", "val_precision", _fmt4),
    ("val_recall", "val_recall", _fmt4),
    ("val_f1", "val_f1", _fmt4),
    ("val_auc", "val_roc_auc", _fmt4),
]


def format_summary_table(ranked: Sequence[dict]) -> str:
    """Fixed-width comparison table, winner marked, followed by the selection rule."""
    cells = [[fmt(s[key]) for _, key, fmt in SUMMARY_COLUMNS] for s in ranked]
    headers = [h for h, _, _ in SUMMARY_COLUMNS]
    widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]

    def line(values):
        return "  ".join(v.ljust(w) for v, w in zip(values, widths)).rstrip()

    lines = [line(headers), line(["-" * w for w in widths])]
    for s, row in zip(ranked, cells):
        lines.append(line(row) + ("   <- winner" if s["rank"] == 1 else ""))
    winner = ranked[0]
    lines += [
        "",
        f"Selection criterion: {SELECTION_CRITERION}",
        f"Winner: {winner['name']} (val_loss {_fmt4(winner['val_loss'])})",
        "Test set: not evaluated.",
    ]
    return "\n".join(lines)


def write_summary(json_path: Path, csv_path: Path, ranked: Sequence[dict],
                  controls: Dict[str, object]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_criterion": SELECTION_CRITERION,
        "winner": ranked[0]["name"],
        "controlled": controls,
        "varied": list(VARIED_FIELDS),
        "test_set_evaluated": False,
        "runs": list(ranked),
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    table = pd.DataFrame(ranked)
    table["hidden_sizes"] = table["hidden_sizes"].map(lambda v: "-".join(str(h) for h in v))
    table.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train several configs under matched controls and compare them on validation.")
    ap.add_argument("configs", nargs="+", help="two or more YAML files in api/configs/")
    ap.add_argument("--output-dir", default=str(MODELS_DIR),
                    help="runs go to <output-dir>/<config name>/ (default: models/)")
    ap.add_argument("--summary-name", default="controlled_comparison",
                    help="summary file name under <output-dir>/experiments/")
    ap.add_argument("--overwrite", action="store_true",
                    help="retrain and replace existing runs and the existing summary")
    ap.add_argument("--reuse-existing", action="store_true",
                    help="use an existing models/<name>/run.json unchanged if it was "
                         "trained from the same config")
    # Deliberately no --evaluate-test and no --write-run: selection uses validation only.
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.overwrite and args.reuse_existing:
        ap.error("--overwrite and --reuse-existing cannot be combined")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.summary_name):
        ap.error("--summary-name must use only letters, digits, '-' or '_'")

    output_dir = Path(args.output_dir)
    summary_json = output_dir / SUMMARY_DIRNAME / f"{args.summary_name}.json"
    summary_csv = summary_json.with_suffix(".csv")
    try:
        configs = load_experiment_configs(args.configs)
        controls = check_controls(configs)
        if (summary_json.exists() or summary_csv.exists()) and not args.overwrite:
            raise ValueError(f"{summary_json} already exists. Pass --overwrite or a "
                             "different --summary-name.")
        plan = plan_runs(configs, output_dir, args.overwrite, args.reuse_existing)
    except ValueError as e:
        raise SystemExit(str(e))

    print(f"[controls] {controls}")
    print(f"[plan] {plan}")

    data = None
    if "train" in plan.values():
        df = load_adult_income()
        print(f"[data] {len(df)} rows; split counts: {df['split'].value_counts().to_dict()}")
        data = prepare_splits(df)  # fit on train rows once, shared by every run
        print(f"[preprocess] fit on {len(data.y['train'])} train rows "
              f"-> {data.X['train'].shape[1]} features")

    rows = run_experiments(
        configs, plan, output_dir,
        X_train=data.X["train"] if data else None,
        y_train=data.y["train"] if data else None,
        X_val=data.X["val"] if data else None,
        y_val=data.y["val"] if data else None,
        preprocessor=data.preprocessor if data else None,
    )

    ranked = rank_runs([summarize_run(r) for r in rows])
    print()
    print(format_summary_table(ranked))
    write_summary(summary_json, summary_csv, ranked, controls)
    print(f"\n[saved] {_repo_relative(summary_json)}, {_repo_relative(summary_csv)}")
    print("[test] not evaluated (final test evaluation is a separate step)")
    print("[runs] not written to Supabase")


if __name__ == "__main__":
    sys.exit(main())
