"""Probability calibration for the Adult Income MLP: temperature scaling.

Kept free of training code (like api/model.py) so the FastAPI service can load
the fitted calibrator next to the checkpoint and the preprocessor:

    P(>50K) = sigmoid(logit / T)

One scalar T > 0 is fit by minimizing binary cross-entropy (NLL) on the
VALIDATION split only. Dividing a logit by T > 0 never changes its sign, so the
0.5 decision boundary, every threshold-0.5 prediction and the ROC-AUC are
unchanged; only how confident the probabilities are moves.

The calibrator is saved as plain JSON (not pickle), so loading it can't run code.

Also holds the calibration metrics: Brier score, expected calibration error
(ECE) and equal-width reliability-bin data for a reliability diagram.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

METHOD = "temperature"
FIT_SPLIT = "val"            # the only split a calibrator is ever fit on
DECISION_THRESHOLD = 0.5     # fixed project-wide; never tuned
DEFAULT_N_BINS = 10          # equal-width reliability bins on [0, 1]
CALIBRATOR_FILENAME = "calibrator.json"  # saved next to model.pt
FORMAT_VERSION = 1

# Search log(T) in a generous range; real temperatures are close to 1.
_LOG_T_BOUNDS = (math.log(0.05), math.log(20.0))

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------
def _labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a non-empty 1-D array")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y.astype(np.float64)


def _logits_labels(logits, y) -> Tuple[np.ndarray, np.ndarray]:
    z = np.asarray(logits, dtype=np.float64)
    y = _labels(y)
    if z.shape != y.shape:
        raise ValueError(f"logits {z.shape} and labels {y.shape} differ in shape")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    return z, y


def _probs_labels(y, prob) -> Tuple[np.ndarray, np.ndarray]:
    y = _labels(y)
    p = np.asarray(prob, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"probabilities {p.shape} and labels {y.shape} differ in shape")
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return y, p


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------
def binary_nll(logits, y) -> float:
    """Mean binary cross-entropy of sigmoid(logits) vs 0/1 labels (stable form)."""
    z, y = _logits_labels(logits, y)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class TemperatureScaler:
    """sigmoid(logit / T) with one fitted T > 0.

    ``fit`` may be called once; after that the calibrator is frozen, so nothing
    downstream (test evaluation, permutation importance, the API) can refit it.
    """

    def __init__(self, temperature: Optional[float] = None, n_fit: Optional[int] = None,
                 nll_before: Optional[float] = None, nll_after: Optional[float] = None):
        if temperature is not None and not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
        self.temperature = None if temperature is None else float(temperature)
        self.n_fit = n_fit
        self.nll_before = nll_before
        self.nll_after = nll_after

    @property
    def is_fitted(self) -> bool:
        return self.temperature is not None

    def fit(self, logits, y) -> "TemperatureScaler":
        """Fit T by minimizing NLL. Pass VALIDATION logits and labels only."""
        if self.is_fitted:
            raise RuntimeError("calibrator is already fitted and frozen; build a new one to refit")
        z, y = _logits_labels(logits, y)
        res = minimize_scalar(
            lambda log_t: binary_nll(z / math.exp(log_t), y),
            bounds=_LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-6},
        )
        self.temperature = float(math.exp(res.x))
        self.n_fit = int(len(y))
        self.nll_before = binary_nll(z, y)
        self.nll_after = binary_nll(z / self.temperature, y)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("calibrator is not fitted; fit it on validation data first")

    def calibrate_logits(self, logits) -> np.ndarray:
        self._require_fitted()
        return np.asarray(logits, dtype=np.float64) / self.temperature

    def predict_proba(self, logits) -> np.ndarray:
        """Calibrated P(>50K) for raw MLP logits."""
        return expit(self.calibrate_logits(logits))

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        self._require_fitted()
        return {
            "method": METHOD,
            "format_version": FORMAT_VERSION,
            "temperature": self.temperature,
            "fitted_on_split": FIT_SPLIT,
            "n_fit": self.n_fit,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "decision_threshold": DECISION_THRESHOLD,
            "formula": "p = sigmoid(logit / temperature)",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemperatureScaler":
        if d.get("method") != METHOD:
            raise ValueError(f"expected a {METHOD!r} calibrator, got {d.get('method')!r}")
        if d.get("fitted_on_split") != FIT_SPLIT:
            raise ValueError(f"calibrator must be fit on {FIT_SPLIT!r}, got {d.get('fitted_on_split')!r}")
        if d.get("temperature") is None:
            raise ValueError("calibrator file has no temperature")
        return cls(d["temperature"], d.get("n_fit"), d.get("nll_before"), d.get("nll_after"))

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False))
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TemperatureScaler":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_calibrator(path: PathLike) -> TemperatureScaler:
    """Load a saved calibrator (for the API, next to model.pt and preprocessor.joblib)."""
    return TemperatureScaler.load(path)


# ---------------------------------------------------------------------------
# calibration metrics
# ---------------------------------------------------------------------------
def brier_score(y, prob) -> float:
    """Mean squared error between P(>50K) and the 0/1 label (lower is better)."""
    y, p = _probs_labels(y, prob)
    return float(np.mean((p - y) ** 2))


def reliability_bins(y, prob, n_bins: int = DEFAULT_N_BINS) -> List[dict]:
    """Equal-width bins on [0, 1] with the data for a reliability diagram.

    Bin b holds lower <= p < upper; the last bin also holds p == 1. Empty bins
    are kept (count 0, means None) so every plot has the same x-axis.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y, p = _probs_labels(y, prob)
    idx = np.minimum(np.floor(p * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append({
            "bin": b,
            "lower": round(b / n_bins, 10),
            "upper": round((b + 1) / n_bins, 10),
            "count": count,
            "mean_predicted": float(p[mask].mean()) if count else None,
            "fraction_positive": float(y[mask].mean()) if count else None,
        })
    return bins


def ece_from_bins(bins: List[dict]) -> float:
    """Count-weighted mean |fraction_positive - mean_predicted| over non-empty bins."""
    n = sum(b["count"] for b in bins)
    if n == 0:
        raise ValueError("no rows in any bin")
    return float(sum(b["count"] / n * abs(b["fraction_positive"] - b["mean_predicted"])
                     for b in bins if b["count"]))


def expected_calibration_error(y, prob, n_bins: int = DEFAULT_N_BINS) -> float:
    return ece_from_bins(reliability_bins(y, prob, n_bins))


def calibration_summary(y, prob_before, prob_after, n_bins: int = DEFAULT_N_BINS) -> dict:
    """Brier, ECE and reliability bins before and after calibration, on one split."""
    bins_before = reliability_bins(y, prob_before, n_bins)
    bins_after = reliability_bins(y, prob_after, n_bins)
    return {
        "n": int(len(np.asarray(y))),
        "n_bins": n_bins,
        "binning": "uniform",
        "brier_before": brier_score(y, prob_before),
        "brier_after": brier_score(y, prob_after),
        "ece_before": ece_from_bins(bins_before),
        "ece_after": ece_from_bins(bins_after),
        "bins_before": bins_before,
        "bins_after": bins_after,
    }
