"""Numerical test of the OBSOLETE starter trainer (api/training.py, synthetic data).

The API no longer uses api/training.py or shared/data.py; the served model comes
from api/train.py + api/evaluate_final.py. This test (and those two modules) can
be deleted together.
"""
from __future__ import annotations

from shared.data import generate_tabular
from api.training import train_income_classifier


def test_classifier_recovers_signal_unit():
    records, labels = generate_tabular(n_rows=2000, noise=1.0, seed=1)
    metrics, model_b64, loss = train_income_classifier(
        records, labels, hidden_dim=32, lr=0.01, batch_size=64, epochs=150
    )
    assert metrics["accuracy"] > 0.8
    assert metrics["roc_auc"] > 0.85
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0
