"""B2 Task 7: XGBoost action classifier training and evaluation.

Trains a multiclass XGBoost classifier on the 120-dim feature vectors produced
by features.py to detect boxing punch types (6 classes from BoxingVI).

Workflow:
    1. Build feature DataFrame: build_feature_dataframe() [features.py]
    2. Split: make_splits() [dataset.py] → train/val/test CSVs
    3. Train: train_classifier(X_train, y_train)
    4. Evaluate: evaluate_classifier(model, X_test, y_test)
    5. Save: save_model(model, path)

Baseline models (random forest, SVM) available for comparison via
train_baseline_classifier().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Public API ────────────────────────────────────────────────────────────────

#: Default XGBoost hyperparameters (validated against synthetic data).
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> Any:
    """Train an XGBoost multiclass classifier.

    Args:
        X_train: (n_samples, n_features) float32 feature matrix.
        y_train: (n_samples,) integer class labels.
        params: XGBoost hyperparameters. Defaults to DEFAULT_XGB_PARAMS.
        X_val: Optional validation features for early stopping.
        y_val: Optional validation labels for early stopping.

    Returns:
        Trained XGBClassifier instance.

    Raises:
        ValueError: If X_train and y_train have mismatched lengths.
    """
    from xgboost import XGBClassifier

    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) must have same length"
        )

    effective_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    # num_class must match number of unique labels
    n_classes = len(np.unique(y_train))
    effective_params["num_class"] = n_classes

    model = XGBClassifier(**effective_params)

    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def train_baseline_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
) -> Any:
    """Train a baseline classifier for comparison with XGBoost.

    Args:
        X_train: (n_samples, n_features) feature matrix.
        y_train: (n_samples,) integer class labels.
        model_type: "random_forest" or "svm".

    Returns:
        Trained sklearn classifier instance.

    Raises:
        ValueError: If model_type is unrecognized.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    elif model_type == "svm":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ])
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Use 'random_forest' or 'svm'.")
    return model


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained classifier and return structured metrics.

    Args:
        model: Trained classifier with a predict() method.
        X_test: (n_samples, n_features) feature matrix.
        y_test: (n_samples,) true integer labels.
        class_names: Optional list of class name strings for the report.

    Returns:
        Dict with keys:
        - "accuracy": float overall accuracy
        - "per_class": dict[class_name, {precision, recall, f1, support}]
        - "confusion_matrix": 2D list (n_classes × n_classes)
        - "classification_report": string report from sklearn
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = model.predict(X_test)

    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    target_names = class_names or [str(i) for i in labels]

    report_str = classification_report(
        y_test, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    per_class: dict[str, dict] = {}
    for name in target_names:
        if name in report_dict:
            per_class[name] = {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1": report_dict[name]["f1-score"],
                "support": int(report_dict[name]["support"]),
            }

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }


def cross_validate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any] | None = None,
    cv: int = 5,
) -> dict[str, Any]:
    """Stratified k-fold cross-validation of the XGBoost classifier.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) integer class labels.
        params: XGBoost hyperparameters. Defaults to DEFAULT_XGB_PARAMS.
        cv: Number of folds (default 5).

    Returns:
        Dict with "scores" (list of per-fold accuracy) and "mean"/"std".
    """
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    effective_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    n_classes = len(np.unique(y))
    effective_params["num_class"] = n_classes

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        model = XGBClassifier(**effective_params)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        acc = float((model.predict(X[val_idx]) == y[val_idx]).mean())
        scores.append(acc)

    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "cv": cv,
    }


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance_df(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importances from a trained XGBoost model as a DataFrame.

    Args:
        model: Trained XGBClassifier with feature_importances_ attribute.
        feature_names: List of feature name strings.

    Returns:
        DataFrame with columns [feature, importance], sorted descending by importance.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model: Any, path: str | Path, metadata: dict | None = None) -> None:
    """Save a trained model and optional metadata to disk.

    XGBoost models are saved in JSON format for portability.
    Metadata (hyperparams, evaluation metrics, version) is saved as a sidecar JSON.

    Args:
        model: Trained XGBClassifier.
        path: Output path (e.g., "models/classifier_v1.json").
        metadata: Optional dict to save alongside the model.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    if metadata is not None:
        meta_path = out.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_model(path: str | Path) -> Any:
    """Load a saved XGBoost model from disk.

    Args:
        path: Path to the saved model JSON file.

    Returns:
        Loaded XGBClassifier ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    from xgboost import XGBClassifier

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = XGBClassifier()
    model.load_model(str(p))
    return model


def load_model_metadata(path: str | Path) -> dict:
    """Load metadata sidecar for a saved model.

    Args:
        path: Path to the model JSON file (not the .meta.json file).

    Returns:
        Metadata dict, or empty dict if no sidecar exists.
    """
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)
