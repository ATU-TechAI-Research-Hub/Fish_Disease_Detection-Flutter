"""Fast unit tests for research-oriented training/evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.evaluate import (  # noqa: E402
    _expected_calibration_error,
    _multiclass_brier_score,
    _top_k_accuracy,
)
from app.core.preprocessing import preprocess_image  # noqa: E402
from train.data import (  # noqa: E402
    discover_training_split,
    make_image_sequence,
)
from train.train import _compute_class_weights  # noqa: E402


def test_effective_number_weights_are_normalized_and_help_minority():
    generator = SimpleNamespace(
        classes=np.asarray([0] * 100 + [1] * 20 + [2] * 5)
    )
    weights = _compute_class_weights(
        generator, num_classes=3, strategy="effective_number", beta=0.99
    )

    assert np.mean(list(weights.values())) == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_inverse_frequency_weights_match_expected_formula():
    generator = SimpleNamespace(classes=np.asarray([0] * 6 + [1] * 3))
    weights = _compute_class_weights(
        generator, num_classes=2, strategy="inverse_frequency"
    )
    assert weights[0] == pytest.approx(0.75)
    assert weights[1] == pytest.approx(1.5)


def test_perfect_predictions_have_zero_brier_and_ece():
    probabilities = np.eye(3, dtype=np.float32)
    labels = np.asarray([0, 1, 2])

    assert _multiclass_brier_score(probabilities, labels, 3) == pytest.approx(0)
    assert _expected_calibration_error(probabilities, labels) == pytest.approx(0)


def test_overconfident_wrong_predictions_have_large_calibration_error():
    probabilities = np.asarray(
        [[0.99, 0.005, 0.005], [0.005, 0.99, 0.005]], dtype=np.float32
    )
    labels = np.asarray([1, 0])

    assert _expected_calibration_error(probabilities, labels) > 0.9
    assert _multiclass_brier_score(probabilities, labels, 3) > 1.9


def test_top_three_accuracy():
    probabilities = np.asarray(
        [
            [0.4, 0.3, 0.2, 0.1],
            [0.05, 0.10, 0.15, 0.70],
        ]
    )
    labels = np.asarray([2, 0])
    assert _top_k_accuracy(probabilities, labels, k=3) == pytest.approx(0.5)


def test_training_sequence_exactly_matches_inference_preprocessing(tmp_path):
    image_path = tmp_path / "portrait.png"
    Image.new("RGBA", (320, 180), (20, 80, 160, 180)).save(image_path)
    sequence = make_image_sequence(
        file_paths=[image_path],
        labels=np.asarray([0]),
        num_classes=2,
        image_size=150,
        batch_size=1,
        shuffle=False,
        seed=42,
        augmenter=None,
    )

    train_tensor, train_target = sequence[0]
    inference_tensor = preprocess_image(image_path, image_size=150)
    np.testing.assert_array_equal(train_tensor, inference_tensor)
    np.testing.assert_array_equal(
        train_target, np.asarray([[1.0, 0.0]], dtype=np.float32)
    )


def test_training_split_is_deterministic_and_stratified(tmp_path):
    for folder in ("class_a", "class_b"):
        class_dir = tmp_path / folder
        class_dir.mkdir()
        for index in range(10):
            (class_dir / f"{index}.jpg").touch()

    first = discover_training_split(
        tmp_path, ["class_a", "class_b"], validation_split=0.2, seed=42
    )
    second = discover_training_split(
        tmp_path, ["class_a", "class_b"], validation_split=0.2, seed=42
    )

    assert first[0] == second[0]
    assert first[2] == second[2]
    assert np.bincount(first[1]).tolist() == [8, 8]
    assert np.bincount(first[3]).tolist() == [2, 2]
