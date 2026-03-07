"""Tests for B2 Task 7: XGBoost action classifier training and evaluation."""

import numpy as np
import pytest

from track_b.b2.classifier import (
    DEFAULT_XGB_PARAMS,
    cross_validate_classifier,
    evaluate_classifier,
    feature_importance_df,
    load_model,
    load_model_metadata,
    save_model,
    train_baseline_classifier,
    train_classifier,
)
from track_b.b2.features import FEATURE_DIM


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_CLASSES = 6
N_TRAIN = 120   # 20 per class — enough to overfit for testing
N_TEST = 30     # 5 per class


def make_synthetic_dataset(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    n_classes: int = N_CLASSES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate separable synthetic feature data for classifier testing.

    Each class is drawn from a Gaussian centred at a distinct point,
    so the model should easily achieve high accuracy.

    Returns:
        (X_train, y_train, X_test, y_test) arrays.
    """
    rng = np.random.default_rng(seed)
    per_class_train = n_train // n_classes
    per_class_test = n_test // n_classes

    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []

    for cls in range(n_classes):
        center = np.zeros(FEATURE_DIM, dtype=np.float32)
        center[cls * 20 : cls * 20 + 20] = float(cls + 1) * 10.0  # well separated
        X_tr = rng.normal(center, 0.1, size=(per_class_train, FEATURE_DIM)).astype(np.float32)
        X_te = rng.normal(center, 0.1, size=(per_class_test, FEATURE_DIM)).astype(np.float32)
        X_train_parts.append(X_tr)
        y_train_parts.append(np.full(per_class_train, cls, dtype=np.int32))
        X_test_parts.append(X_te)
        y_test_parts.append(np.full(per_class_test, cls, dtype=np.int32))

    return (
        np.concatenate(X_train_parts),
        np.concatenate(y_train_parts),
        np.concatenate(X_test_parts),
        np.concatenate(y_test_parts),
    )


@pytest.fixture(scope="module")
def synthetic_data():
    return make_synthetic_dataset()


@pytest.fixture(scope="module")
def trained_model(synthetic_data):
    X_train, y_train, _, _ = synthetic_data
    return train_classifier(X_train, y_train)


# ── DEFAULT_XGB_PARAMS ────────────────────────────────────────────────────────

class TestDefaultParams:
    def test_has_required_keys(self):
        for key in ("objective", "n_estimators", "max_depth", "learning_rate"):
            assert key in DEFAULT_XGB_PARAMS

    def test_objective_is_multiclass(self):
        assert "multi" in DEFAULT_XGB_PARAMS["objective"]


# ── train_classifier ──────────────────────────────────────────────────────────

class TestTrainClassifier:
    def test_returns_model(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_classifier(X_train, y_train)
        assert hasattr(model, "predict")

    def test_model_can_predict(self, trained_model, synthetic_data):
        _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_predictions_are_valid_classes(self, trained_model, synthetic_data):
        _, _, X_test, _ = synthetic_data
        preds = trained_model.predict(X_test)
        assert set(preds.tolist()).issubset(set(range(N_CLASSES)))

    def test_high_accuracy_on_separable_data(self, trained_model, synthetic_data):
        """Separable synthetic data should yield near-perfect accuracy."""
        _, _, X_test, y_test = synthetic_data
        preds = trained_model.predict(X_test)
        accuracy = (preds == y_test).mean()
        assert accuracy >= 0.95

    def test_raises_on_length_mismatch(self):
        X = np.zeros((10, FEATURE_DIM), dtype=np.float32)
        y = np.zeros(5, dtype=np.int32)
        with pytest.raises(ValueError, match="same length"):
            train_classifier(X, y)

    def test_custom_params_applied(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_classifier(X_train, y_train, params={"n_estimators": 5})
        assert model.n_estimators == 5

    def test_with_validation_set(self, synthetic_data):
        X_train, y_train, X_val, y_val = synthetic_data
        model = train_classifier(X_train, y_train, X_val=X_val, y_val=y_val)
        assert hasattr(model, "predict")

    def test_predict_proba_shape(self, trained_model, synthetic_data):
        _, _, X_test, _ = synthetic_data
        proba = trained_model.predict_proba(X_test)
        assert proba.shape == (len(X_test), N_CLASSES)

    def test_predict_proba_sums_to_one(self, trained_model, synthetic_data):
        _, _, X_test, _ = synthetic_data
        proba = trained_model.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ── train_baseline_classifier ─────────────────────────────────────────────────

class TestTrainBaselineClassifier:
    def test_random_forest_returns_model(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_baseline_classifier(X_train, y_train, "random_forest")
        assert hasattr(model, "predict")

    def test_svm_returns_model(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_baseline_classifier(X_train, y_train, "svm")
        assert hasattr(model, "predict")

    def test_random_forest_high_accuracy(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_baseline_classifier(X_train, y_train, "random_forest")
        preds = model.predict(X_test)
        assert (preds == y_test).mean() >= 0.90

    def test_raises_on_unknown_model_type(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_baseline_classifier(X_train, y_train, "neural_network")


# ── evaluate_classifier ───────────────────────────────────────────────────────

class TestEvaluateClassifier:
    def test_returns_expected_keys(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        assert {"accuracy", "per_class", "confusion_matrix", "classification_report"}.issubset(
            result.keys()
        )

    def test_accuracy_in_valid_range(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_high_accuracy_on_separable_data(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        assert result["accuracy"] >= 0.95

    def test_per_class_has_all_classes_with_names(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        class_names = ["jab", "cross", "lead_hook", "rear_hook", "lead_uppercut", "rear_uppercut"]
        result = evaluate_classifier(trained_model, X_test, y_test, class_names=class_names)
        for name in class_names:
            assert name in result["per_class"]

    def test_per_class_metrics_in_valid_range(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        for metrics in result["per_class"].values():
            assert 0.0 <= metrics["precision"] <= 1.0
            assert 0.0 <= metrics["recall"] <= 1.0
            assert 0.0 <= metrics["f1"] <= 1.0
            assert metrics["support"] >= 0

    def test_confusion_matrix_shape(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        cm = result["confusion_matrix"]
        assert len(cm) == N_CLASSES
        assert all(len(row) == N_CLASSES for row in cm)

    def test_confusion_matrix_row_sums(self, trained_model, synthetic_data):
        """Each row of confusion matrix should sum to the true class count."""
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        cm = np.array(result["confusion_matrix"])
        # Row sums = number of true samples per class
        for i, row_sum in enumerate(cm.sum(axis=1)):
            expected = int((y_test == i).sum())
            assert row_sum == expected

    def test_classification_report_is_string(self, trained_model, synthetic_data):
        _, _, X_test, y_test = synthetic_data
        result = evaluate_classifier(trained_model, X_test, y_test)
        assert isinstance(result["classification_report"], str)


# ── cross_validate_classifier ─────────────────────────────────────────────────

class TestCrossValidateClassifier:
    def test_returns_expected_keys(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        result = cross_validate_classifier(X_train, y_train, cv=3)
        assert {"scores", "mean", "std", "cv"}.issubset(result.keys())

    def test_correct_number_of_folds(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        result = cross_validate_classifier(X_train, y_train, cv=3)
        assert len(result["scores"]) == 3
        assert result["cv"] == 3

    def test_scores_in_valid_range(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        result = cross_validate_classifier(X_train, y_train, cv=3)
        for score in result["scores"]:
            assert 0.0 <= score <= 1.0

    def test_high_mean_accuracy_on_separable_data(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        result = cross_validate_classifier(X_train, y_train, cv=3)
        assert result["mean"] >= 0.85

    def test_mean_std_consistent(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        result = cross_validate_classifier(X_train, y_train, cv=3)
        np.testing.assert_allclose(result["mean"], np.mean(result["scores"]), atol=1e-6)
        np.testing.assert_allclose(result["std"], np.std(result["scores"]), atol=1e-6)


# ── feature_importance_df ─────────────────────────────────────────────────────

class TestFeatureImportanceDf:
    def test_returns_dataframe(self, trained_model):
        from track_b.b2.features import feature_names
        df = feature_importance_df(trained_model, feature_names())
        import pandas as pd
        assert isinstance(df, pd.DataFrame)

    def test_has_feature_and_importance_columns(self, trained_model):
        from track_b.b2.features import feature_names
        df = feature_importance_df(trained_model, feature_names())
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_sorted_descending(self, trained_model):
        from track_b.b2.features import feature_names
        df = feature_importance_df(trained_model, feature_names())
        assert df["importance"].is_monotonic_decreasing

    def test_importance_sums_to_one(self, trained_model):
        from track_b.b2.features import feature_names
        df = feature_importance_df(trained_model, feature_names())
        np.testing.assert_allclose(df["importance"].sum(), 1.0, atol=1e-5)

    def test_row_count_matches_feature_dim(self, trained_model):
        from track_b.b2.features import FEATURE_DIM, feature_names
        df = feature_importance_df(trained_model, feature_names())
        assert len(df) == FEATURE_DIM


# ── save_model / load_model ───────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_creates_file(self, trained_model, tmp_path):
        path = tmp_path / "model.json"
        save_model(trained_model, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, trained_model, tmp_path):
        path = tmp_path / "deep" / "nested" / "model.json"
        save_model(trained_model, path)
        assert path.exists()

    def test_save_metadata_creates_sidecar(self, trained_model, tmp_path):
        path = tmp_path / "model.json"
        meta = {"version": "v1", "accuracy": 0.95}
        save_model(trained_model, path, metadata=meta)
        assert (tmp_path / "model.meta.json").exists()

    def test_load_model_returns_model(self, trained_model, tmp_path):
        path = tmp_path / "model.json"
        save_model(trained_model, path)
        loaded = load_model(path)
        assert hasattr(loaded, "predict")

    def test_loaded_model_matches_original(self, trained_model, synthetic_data, tmp_path):
        """Loaded model should give identical predictions to the original."""
        _, _, X_test, _ = synthetic_data
        path = tmp_path / "model.json"
        save_model(trained_model, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(
            trained_model.predict(X_test),
            loaded.predict(X_test),
        )

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.json")

    def test_load_metadata_returns_dict(self, trained_model, tmp_path):
        path = tmp_path / "model.json"
        meta = {"version": "v1", "n_classes": 6}
        save_model(trained_model, path, metadata=meta)
        loaded_meta = load_model_metadata(path)
        assert loaded_meta["version"] == "v1"
        assert loaded_meta["n_classes"] == 6

    def test_load_metadata_empty_if_no_sidecar(self, trained_model, tmp_path):
        path = tmp_path / "model.json"
        save_model(trained_model, path)  # no metadata
        loaded_meta = load_model_metadata(path)
        assert loaded_meta == {}
