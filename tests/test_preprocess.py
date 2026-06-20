import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from preprocess import preprocess
from food_info import CLASS_NAMES, FOOD_INFO, FOLDER_TO_CLASS


def make_dummy_dataset(n_per_class=10):
    """Build a small synthetic dataset for testing preprocessing logic
    without needing the real Fruits-360 images."""
    images = []
    labels = []
    for cls in CLASS_NAMES:
        for _ in range(n_per_class):
            images.append(np.random.randint(0, 255, size=(100, 100, 3)).astype(np.float32))
            labels.append(cls)
    return np.array(images), np.array(labels)


def test_preprocess_output_shapes(tmp_path):
    """preprocess() should return train/val/test splits whose total
    size matches the input dataset size."""
    X, y = make_dummy_dataset(n_per_class=10)
    encoder_path = str(tmp_path / "label_encoder.pkl")

    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess(
        X, y, save_encoder=True, encoder_path=encoder_path
    )

    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == X.shape[0]


def test_preprocess_split_ratios(tmp_path):
    """Train/val/test split should roughly follow the configured
    70/15/15 ratio (30% held out, then split evenly into val/test)."""
    X, y = make_dummy_dataset(n_per_class=20)
    encoder_path = str(tmp_path / "label_encoder.pkl")

    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess(
        X, y, save_encoder=True, encoder_path=encoder_path
    )

    total = X.shape[0]
    train_ratio = X_train.shape[0] / total
    val_ratio = X_val.shape[0] / total
    test_ratio = X_test.shape[0] / total

    assert abs(train_ratio - 0.70) < 0.05
    assert abs(val_ratio - 0.15) < 0.05
    assert abs(test_ratio - 0.15) < 0.05


def test_preprocess_normalizes_pixel_values(tmp_path):
    """Pixel values should be scaled to the 0-1 range after preprocessing."""
    X, y = make_dummy_dataset(n_per_class=20)
    encoder_path = str(tmp_path / "label_encoder.pkl")

    X_train, _, _, _, _, _, _ = preprocess(X, y, save_encoder=True, encoder_path=encoder_path)

    assert X_train.max() <= 1.0
    assert X_train.min() >= 0.0


def test_preprocess_num_classes_matches_class_names():
    """num_classes returned should match the total number of defined classes."""
    X, y = make_dummy_dataset(n_per_class=20)
    _, _, _, _, _, _, num_classes = preprocess(X, y, save_encoder=False)

    assert num_classes == len(CLASS_NAMES)


def test_label_encoder_round_trip(tmp_path):
    """A saved label encoder should correctly decode predicted indices
    back to the original class name."""
    X, y = make_dummy_dataset(n_per_class=20)
    encoder_path = str(tmp_path / "label_encoder.pkl")

    preprocess(X, y, save_encoder=True, encoder_path=encoder_path)

    import pickle
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    decoded = encoder.inverse_transform(encoder.transform(CLASS_NAMES))
    assert list(decoded) == CLASS_NAMES