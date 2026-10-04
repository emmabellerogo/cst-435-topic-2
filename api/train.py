"""Config-driven training CLI for the Adult Income MLP.

Usage (from the repo root, with a .env holding the service-role key):

    python -m api.train --config api/configs/baseline.yaml
    python -m api.train --config api/configs/baseline.yaml --evaluate-test
    python -m api.train --config api/configs/baseline.yaml --evaluate-test --write-run

Steps:
  1. read the adult_income rows from Supabase (via api.db)
  2. separate them by the fixed ``split`` column assigned by db/load.py
  3. fit the preprocessor on TRAIN rows only, then transform train/val/test
  4. train on train; after every epoch score val and keep the weights with the
     lowest val loss (early stopping on val loss if configured)
  5. save model.pt, preprocessor.joblib, history.json and run.json under
     models/<config name>/
  6. only with --evaluate-test: score the best checkpoint on test, once
  7. only with --write-run: insert run.json as one row in the ``runs`` table

The test split never influences training, early stopping or checkpoint choice.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from torch import nn

from api.model import ACTIVATIONS, MIN_HIDDEN_LAYERS, MLP, save_checkpoint
from api.preprocessing import fit_preprocessor, save_preprocessor, split_xy, transform_features

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """One training run's hyperparameters, loaded from api/configs/*.yaml."""

    name: str                    # also the output folder: models/<name>/
    hidden_sizes: List[int]
    activation: str
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int                  # maximum epoch budget
    batch_size: int
    seed: int
    early_stopping_patience: Optional[int] = None  # None = always run every epoch
    architecture: str = "mlp"

    def __post_init__(self):
        self.dropout = float(self.dropout)
        self.learning_rate = float(self.learning_rate)  # YAML reads "1e-3" as a string
        self.weight_decay = float(self.weight_decay)
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)
        self.activation = str(self.activation).lower()
        if self.early_stopping_patience is not None:
            self.early_stopping_patience = int(self.early_stopping_patience)

        problems = []
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            problems.append("name must use only letters, digits, '-' or '_' (it names the models/ folder)")
        if self.architecture != "mlp":
            problems.append(f"architecture must be 'mlp', got {self.architecture!r}")
        if not isinstance(self.hidden_sizes, (list, tuple)) or len(self.hidden_sizes) < MIN_HIDDEN_LAYERS:
            problems.append(f"hidden_sizes needs at least {MIN_HIDDEN_LAYERS} layers, got {self.hidden_sizes!r}")
        elif any(int(h) <= 0 for h in self.hidden_sizes):
            problems.append(f"hidden_sizes must be positive, got {self.hidden_sizes!r}")
        if self.activation not in ACTIVATIONS:
            problems.append(f"activation must be one of {sorted(ACTIVATIONS)}, got {self.activation!r}")
        if not 0.0 <= self.dropout < 1.0:
            problems.append("dropout must be in [0, 1)")
        if self.learning_rate <= 0:
            problems.append("learning_rate must be > 0")
        if self.weight_decay < 0:
            problems.append("weight_decay must be >= 0")
        if self.epochs < 1:
            problems.append("epochs must be >= 1")
        if self.batch_size < 1:
            problems.append("batch_size must be >= 1")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            problems.append("early_stopping_patience must be >= 1 or null")
        if problems:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(problems))
        self.hidden_sizes = [int(h) for h in self.hidden_sizes]


def load_config(path) -> TrainConfig:
    """Load and validate a YAML config. Unknown or missing keys are errors."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping of hyperparameters")
    all_keys = {f.name for f in fields(TrainConfig)}
    optional = {"early_stopping_patience", "architecture"}
    unknown = sorted(set(raw) - all_keys)
    missing = sorted(all_keys - optional - set(raw))
    if unknown or missing:
        raise ValueError(f"{path}: unknown keys {unknown}, missing keys {missing}")
    return TrainConfig(**raw)


def set_seed(seed: int) -> None:
    """Seed every RNG that affects initialization, shuffling and dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class PreparedData:
    preprocessor: Pipeline
    X: Dict[str, np.ndarray]   # split -> float32 feature matrix
    y: Dict[str, np.ndarray]   # split -> 0/1 income_label
    ids: Dict[str, Optional[np.ndarray]]  # split -> adult_income.id (for later audit logging)


def load_adult_income(split: Optional[str] = None) -> pd.DataFrame:
    """Read the training columns of adult_income rows from Supabase.

    Every row by default; pass ``split`` to read only that split.
    """
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")):
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first.")
    from api import db

    return pd.DataFrame(db.fetch_adult_income(split=split))


def prepare_splits(df: pd.DataFrame) -> PreparedData:
    """Separate rows by the stored ``split`` column; fit the preprocessor on train only."""
    if "split" not in df.columns:
        raise ValueError("rows need the 'split' column")
    if df["split"].isna().any():
        raise ValueError(f"{int(df['split'].isna().sum())} rows have no split assigned")
    unexpected = sorted(set(df["split"].unique()) - set(SPLITS))
    if unexpected:
        raise ValueError(f"unexpected split values: {unexpected}")
    parts = {s: df[df["split"] == s] for s in SPLITS}
    empty = [s for s, part in parts.items() if part.empty]
    if empty:
        raise ValueError(f"no rows in split(s): {empty}")

    pre = fit_preprocessor(parts["train"])  # train rows ONLY
    return PreparedData(
        preprocessor=pre,
        X={s: transform_features(pre, part) for s, part in parts.items()},
        y={s: split_xy(part)[1] for s, part in parts.items()},
        ids={s: part["id"].to_numpy() if "id" in part.columns else None for s, part in parts.items()},
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Optional[float]]:
    y_pred = (prob >= 0.5).astype(int)
    try:
        auc: Optional[float] = float(roc_auc_score(y_true, prob))
    except ValueError:  # only one class present
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


@torch.no_grad()
def evaluate(model: MLP, X: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    """BCE loss + threshold-0.5 classification metrics, in eval mode (no dropout)."""
    model.eval()
    logits = model(torch.as_tensor(X, dtype=torch.float32))
    target = torch.as_tensor(y, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target).item()
    return {"loss": float(loss), **classification_metrics(np.asarray(y), torch.sigmoid(logits).numpy())}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
class BestCheckpoint:
    """Keeps a copy of the weights from the epoch with the lowest validation loss.

    Also drives early stopping: ``should_stop`` becomes true after ``patience``
    consecutive epochs without a strictly lower val loss.
    """

    def __init__(self, patience: Optional[int] = None):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch: Optional[int] = None
        self.best_metrics: Optional[dict] = None
        self.best_state: Optional[dict] = None
        self.epochs_without_improvement = 0

    def update(self, epoch: int, val_metrics: dict, model: nn.Module) -> bool:
        """Record one epoch's val metrics; return True if this is the new best."""
        val_loss = val_metrics["loss"]
        if val_loss < self.best_loss:  # NaN never counts as an improvement
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_metrics = dict(val_metrics)
            self.best_state = copy.deepcopy(model.state_dict())  # a copy, not live weights
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.epochs_without_improvement >= self.patience


@dataclass
class TrainResult:
    model: MLP                 # holds the BEST epoch's weights, in eval mode
    history: List[dict]        # one entry per epoch run, for loss/accuracy curves
    best_epoch: int
    best_val_metrics: dict
    stopped_early: bool


def _fmt(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def train_model(
    cfg: TrainConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train on (X_train, y_train); select the best epoch on (X_val, y_val).

    Takes no test data at all, so test rows cannot influence training.
    """
    set_seed(cfg.seed)
    model = MLP(X_train.shape[1], cfg.hidden_sizes, cfg.activation, cfg.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    shuffle_gen = torch.Generator().manual_seed(cfg.seed)

    tracker = BestCheckpoint(cfg.early_stopping_patience)
    history: List[dict] = []
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=shuffle_gen)
        running = 0.0
        for start in range(0, len(xt), cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)

        train_m = evaluate(model, X_train, y_train)
        val_m = evaluate(model, X_val, y_val)
        improved = tracker.update(epoch, val_m, model)
        history.append({
            "epoch": epoch,
            "train_batch_loss": running / len(xt),  # mean loss during the epoch (dropout on)
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            "improved": improved,
        })
        log(f"epoch {epoch:3d}/{cfg.epochs}  train_loss {_fmt(train_m['loss'])}  "
            f"val_loss {_fmt(val_m['loss'])}  val_acc {_fmt(val_m['accuracy'])}  "
            f"val_auc {_fmt(val_m['roc_auc'])}{'  *best' if improved else ''}")
        if tracker.should_stop:
            stopped_early = True
            log(f"early stopping: no val-loss improvement for {cfg.early_stopping_patience} epochs")
            break

    if tracker.best_state is None:
        raise RuntimeError("validation loss was never finite; no checkpoint to keep")
    model.load_state_dict(tracker.best_state)
    model.eval()
    return TrainResult(model, history, tracker.best_epoch, tracker.best_metrics, stopped_early)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def _repo_relative(path: Path) -> str:
    """Store paths relative to the repo root so they also work on Render."""
    path = Path(path)
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(path)
    return path.as_posix()


def save_artifacts(out_dir: Path, cfg: TrainConfig, result: TrainResult,
                   pre: Pipeline) -> Tuple[Path, Path]:
    """Write model.pt, preprocessor.joblib and history.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_checkpoint(
        out_dir / "model.pt", result.model,
        config=asdict(cfg),
        feature_names=[str(n) for n in pre.get_feature_names_out()],
        best_epoch=result.best_epoch,
        val_metrics=result.best_val_metrics,
    )
    pre_path = save_preprocessor(pre, out_dir / "preprocessor.joblib")
    (out_dir / "history.json").write_text(json.dumps(result.history, indent=2))
    return ckpt_path, pre_path


def build_run_row(
    cfg: TrainConfig,
    result: TrainResult,
    checkpoint_path: Path,
    preprocessor_path: Path,
    test_metrics: Optional[dict] = None,
) -> dict:
    """One row for the Supabase ``runs`` table (column names match 001_init.sql).

    Test columns stay NULL unless ``test_metrics`` is given. Calibration,
    confusion-matrix and permutation-importance columns are filled by later steps.
    """
    best = result.history[result.best_epoch - 1]
    train_m = {k[len("train_"):]: v for k, v in best.items()
               if k.startswith("train_") and k != "train_batch_loss"}
    val_m = result.best_val_metrics
    test_m = test_metrics or {}
    return {
        "name": cfg.name,
        "architecture": cfg.architecture,
        "hidden_sizes": cfg.hidden_sizes,
        "activation": cfg.activation,
        "dropout": cfg.dropout,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "epochs": cfg.epochs,
        "best_epoch": result.best_epoch,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "train_loss": train_m["loss"],
        "val_loss": val_m["loss"],
        "val_accuracy": val_m["accuracy"],
        "val_roc_auc": val_m["roc_auc"],
        "test_accuracy": test_m.get("accuracy"),
        "test_precision": test_m.get("precision"),
        "test_recall": test_m.get("recall"),
        "test_f1": test_m.get("f1"),
        "test_roc_auc": test_m.get("roc_auc"),
        "train_metrics": train_m,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "checkpoint_path": _repo_relative(checkpoint_path),
        "preprocessor_path": _repo_relative(preprocessor_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train the Adult Income MLP from a YAML config.")
    ap.add_argument("--config", required=True, help="path to a YAML file in api/configs/")
    ap.add_argument("--output-dir", default=str(MODELS_DIR),
                    help="artifacts go to <output-dir>/<config name>/ (default: models/)")
    ap.add_argument("--evaluate-test", action="store_true",
                    help="after training, score the best checkpoint on the test split once")
    ap.add_argument("--write-run", action="store_true",
                    help="insert the run row into the Supabase runs table")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing artifacts for this config name")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) / cfg.name
    if (out_dir / "model.pt").exists() and not args.overwrite:
        raise SystemExit(f"{out_dir / 'model.pt'} already exists. Use --overwrite to replace it.")
    print(f"[config] {args.config}: {asdict(cfg)}")

    df = load_adult_income()
    print(f"[data] {len(df)} rows; split counts: {df['split'].value_counts().to_dict()}")
    data = prepare_splits(df)
    print(f"[preprocess] fit on {len(data.y['train'])} train rows -> {data.X['train'].shape[1]} features")

    result = train_model(cfg, data.X["train"], data.y["train"], data.X["val"], data.y["val"])
    print(f"[best] epoch {result.best_epoch}: {result.best_val_metrics}")
    ckpt_path, pre_path = save_artifacts(out_dir, cfg, result, data.preprocessor)

    test_metrics = None
    if args.evaluate_test:
        test_metrics = evaluate(result.model, data.X["test"], data.y["test"])
        print(f"[test] {test_metrics}")
    else:
        print("[test] not evaluated (pass --evaluate-test for the final evaluation)")

    row = build_run_row(cfg, result, ckpt_path, pre_path, test_metrics)
    (out_dir / "run.json").write_text(json.dumps(row, indent=2, allow_nan=False))
    print(f"[saved] {_repo_relative(out_dir)}/: model.pt, preprocessor.joblib, history.json, run.json")

    if args.write_run:
        from api import db

        saved = db.insert_training_run(row)
        print(f"[runs] inserted run id={saved['id']}")
    else:
        print("[runs] not written to Supabase (pass --write-run)")


if __name__ == "__main__":
    sys.exit(main())
