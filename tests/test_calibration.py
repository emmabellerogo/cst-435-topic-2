"""Offline tests for temperature scaling and the calibration metrics.

Hand-computed values and synthetic logits only -- no Supabase, no real data.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from api.calibration import (
    DECISION_THRESHOLD,
    TemperatureScaler,
    binary_nll,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    load_calibrator,
    reliability_bins,
)


def _overconfident_logits(true_t=2.5, n=20000, seed=0):
    """Logits whose real probabilities are sigmoid(z / true_t): over-confident by true_t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 3.0, n)
    y = (rng.random(n) < expit(z / true_t)).astype(int)
    return z, y


# ---------------------------------------------------------------------------
# Brier / ECE / bins
# ---------------------------------------------------------------------------
def test_brier_score_hand_computed():
    y = [0, 1, 1, 0]
    p = [0.1, 0.8, 0.6, 0.3]
    # (0.01 + 0.04 + 0.16 + 0.09) / 4
    assert brier_score(y, p) == pytest.approx(0.075)


def test_reliability_bins_and_ece_hand_computed():
    y = [0, 1, 1, 1]
    p = [0.2, 0.4, 0.6, 0.9]
    bins = reliability_bins(y, p, n_bins=2)
    assert [b["count"] for b in bins] == [2, 2]
    assert bins[0]["mean_predicted"] == pytest.approx(0.3)
    assert bins[0]["fraction_positive"] == pytest.approx(0.5)
    assert bins[1]["mean_predicted"] == pytest.approx(0.75)
    assert bins[1]["fraction_positive"] == pytest.approx(1.0)
    assert (bins[0]["lower"], bins[0]["upper"], bins[1]["upper"]) == (0.0, 0.5, 1.0)
    # 2/4 * |0.5 - 0.3| + 2/4 * |1.0 - 0.75|
    assert expected_calibration_error(y, p, n_bins=2) == pytest.approx(0.225)


def test_bin_edges_and_empty_bins():
    bins = reliability_bins([0, 1, 1], [0.0, 0.5, 1.0], n_bins=4)
    assert [b["count"] for b in bins] == [1, 0, 1, 1]  # 0.5 -> bin 2; 1.0 -> last bin
    assert bins[1]["mean_predicted"] is None and bins[1]["fraction_positive"] is None
    json.dumps(bins, allow_nan=False)  # JSON-friendly for runs.calibration_bins


def test_perfect_confident_predictions_have_zero_error():
    y = [0, 0, 1, 1]
    p = [0.0, 0.0, 1.0, 1.0]
    assert brier_score(y, p) == 0.0
    assert expected_calibration_error(y, p) == 0.0


@pytest.mark.parametrize("y, p", [
    ([0, 1], [0.5, 1.2]),     # probability > 1
    ([0, 1], [0.5]),          # length mismatch
    ([0, 2], [0.5, 0.5]),     # label not 0/1
    ([], []),                 # empty
])
def test_metric_inputs_are_validated(y, p):
    with pytest.raises(ValueError):
        brier_score(y, p)


def test_calibration_summary_contents():
    y = [0, 1, 1, 0]
    s = calibration_summary(y, [0.1, 0.9, 0.8, 0.4], [0.2, 0.8, 0.7, 0.4], n_bins=5)
    assert s["n"] == 4 and s["n_bins"] == 5
    assert set(s) >= {"brier_before", "brier_after", "ece_before", "ece_after",
                      "bins_before", "bins_after"}
    assert sum(b["count"] for b in s["bins_after"]) == 4


# ---------------------------------------------------------------------------
# temperature scaling
# ---------------------------------------------------------------------------
def test_binary_nll_matches_closed_form():
    z, y = np.array([0.0, 2.0, -1.0]), np.array([1, 1, 0])
    p = expit(z)
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert binary_nll(z, y) == pytest.approx(expected)


def test_temperature_recovers_known_overconfidence():
    z, y = _overconfident_logits(true_t=2.5)
    cal = TemperatureScaler().fit(z, y)
    assert cal.temperature == pytest.approx(2.5, abs=0.15)
    assert cal.nll_after < cal.nll_before
    assert cal.n_fit == len(y)
    before = expected_calibration_error(y, expit(z))
    after = expected_calibration_error(y, cal.predict_proba(z))
    assert after < before


def test_temperature_keeps_threshold_labels_and_ranking():
    z, y = _overconfident_logits()
    cal = TemperatureScaler().fit(z, y)
    raw, calibrated = expit(z), cal.predict_proba(z)
    np.testing.assert_array_equal(raw >= DECISION_THRESHOLD, calibrated >= DECISION_THRESHOLD)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, raw))
    assert DECISION_THRESHOLD == 0.5


def test_calibrator_is_frozen_after_fit():
    z, y = _overconfident_logits(n=500)
    cal = TemperatureScaler().fit(z, y)
    with pytest.raises(RuntimeError, match="frozen"):
        cal.fit(z, y)


def test_unfitted_calibrator_cannot_predict_or_save(tmp_path):
    cal = TemperatureScaler()
    assert not cal.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict_proba([0.0])
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.save(tmp_path / "c.json")


def test_calibrator_serialization_roundtrip(tmp_path):
    z, y = _overconfident_logits(n=2000)
    cal = TemperatureScaler().fit(z, y)
    path = cal.save(tmp_path / "gelu" / "calibrator.json")

    raw = json.loads(path.read_text())  # plain JSON, not a pickle
    assert raw["method"] == "temperature"
    assert raw["fitted_on_split"] == "val"
    assert raw["decision_threshold"] == 0.5

    loaded = load_calibrator(path)
    assert loaded.is_fitted and loaded.temperature == cal.temperature
    assert loaded.n_fit == cal.n_fit
    np.testing.assert_allclose(loaded.predict_proba(z), cal.predict_proba(z))


@pytest.mark.parametrize("change", [
    {"method": "isotonic"},
    {"fitted_on_split": "test"},
    {"temperature": -1.0},
    {"temperature": None},
])
def test_load_rejects_bad_calibrator_files(tmp_path, change):
    z, y = _overconfident_logits(n=500)
    d = {**TemperatureScaler().fit(z, y).to_dict(), **change}
    path = tmp_path / "calibrator.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_calibrator(path)
