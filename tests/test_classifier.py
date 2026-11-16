"""
tests/test_classifier.py
------------------------
Unit tests for the NLP failure mode classifier.
Tests:
  - Text preprocessing
  - Model pipeline loading and prediction
  - Top-k probability distribution structure
"""

import os
import pytest
from src.classifier import (
    load_classifier_pipeline,
    predict_failure_mode,
    preprocess_text,
    MODEL_PATH,
)


def test_preprocess_text():
    sample = "  URGENT RMA: DC-Bus capcitor bulged!! Check temp > 85C...  "
    cleaned = preprocess_text(sample)
    assert "urgent rma" in cleaned
    assert "capcitor bulged" in cleaned
    assert "!" not in cleaned


def test_classifier_inference():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model artifact not yet trained.")

    pipe = load_classifier_pipeline(MODEL_PATH)
    assert pipe is not None

    sample_claim = "Field engineer reported DC bus capacitor venting electrolyte near terminal."
    pred = predict_failure_mode(sample_claim, pipeline=pipe, top_k=3)

    assert "predicted_failure_mode" in pred
    assert "confidence" in pred
    assert "top_k_predictions" in pred
    assert len(pred["top_k_predictions"]) == 3
    assert pred["confidence"] > 0.0
    assert pred["predicted_failure_mode"] == "Capacitor Aging"
