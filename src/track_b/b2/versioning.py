"""B2 Task 9: Model versioning and evaluation log.

Provides tools for tracking model versions and evaluation metrics across
training runs. Supports the iteration loop: train → evaluate → log → compare.

Upgrade path:
    v1: XGBoost on BoxingVI + Olympic Boxing (current)
    v2: AcT (Action Transformer) when combined dataset > ~10K clips
        - evaluate_act_vs_xgboost() documents the comparison
    v3: Own tactical model (B4 extension, requires ~5K interaction sequences)

Usage::

    log = EvaluationLog.load(path)  # or EvaluationLog() for new
    log.record(version="v1", metrics=evaluate_classifier(model, X_test, y_test),
               notes="BoxingVI only, 6 classes")
    log.save(path)

    compare_versions(log, ["v1", "v2"])  # prints side-by-side table
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ModelVersion:
    """Record of one trained model version's evaluation metrics."""

    version: str                         # e.g., "v1", "v1.1", "v2"
    model_type: str                      # "xgboost", "random_forest", "act"
    accuracy: float                      # overall accuracy on test set
    per_class: dict[str, dict]           # {class_name: {precision, recall, f1, support}}
    n_train: int                         # training samples used
    n_test: int                          # test samples evaluated
    dataset: str                         # e.g., "boxingvi_only", "boxingvi+olympic"
    notes: str = ""                      # free-text notes
    model_path: str = ""                 # path to saved model artifact
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def macro_f1(self) -> float:
        """Average F1 across all classes (macro average)."""
        if not self.per_class:
            return 0.0
        return sum(v["f1"] for v in self.per_class.values()) / len(self.per_class)


@dataclass
class EvaluationLog:
    """Ordered log of model version evaluation records."""

    versions: list[ModelVersion] = field(default_factory=list)

    def record(
        self,
        version: str,
        model_type: str,
        metrics: dict[str, Any],
        n_train: int,
        n_test: int,
        dataset: str,
        notes: str = "",
        model_path: str = "",
    ) -> ModelVersion:
        """Add an evaluation record to the log.

        Args:
            version: Version tag (e.g., "v1", "v1.1").
            model_type: Classifier type ("xgboost", "random_forest", "act").
            metrics: Dict from evaluate_classifier() with "accuracy" and "per_class".
            n_train: Number of training samples.
            n_test: Number of test samples.
            dataset: Dataset identifier string.
            notes: Optional free-text notes.
            model_path: Optional path to saved model artifact.

        Returns:
            The created ModelVersion record.
        """
        mv = ModelVersion(
            version=version,
            model_type=model_type,
            accuracy=metrics["accuracy"],
            per_class=metrics.get("per_class", {}),
            n_train=n_train,
            n_test=n_test,
            dataset=dataset,
            notes=notes,
            model_path=model_path,
        )
        self.versions.append(mv)
        return mv

    def get(self, version: str) -> ModelVersion | None:
        """Retrieve a version record by version tag."""
        for mv in self.versions:
            if mv.version == version:
                return mv
        return None

    def best_version(self, metric: str = "accuracy") -> ModelVersion | None:
        """Return the version with the highest value of the given metric.

        Args:
            metric: "accuracy" or "macro_f1" (default "accuracy").

        Returns:
            Best ModelVersion, or None if log is empty.
        """
        if not self.versions:
            return None
        return max(self.versions, key=lambda mv: getattr(mv, metric))

    def is_improvement(self, version: str, metric: str = "accuracy") -> bool:
        """Check if a version improves on all previous versions.

        Args:
            version: Version tag to check.
            metric: Metric to compare ("accuracy" or "macro_f1").

        Returns:
            True if this version has the highest metric value seen so far.
            False if the version is not in the log.
        """
        mv = self.get(version)
        if mv is None:
            return False
        current_value = getattr(mv, metric)
        prior = [v for v in self.versions if v.version != version]
        if not prior:
            return True
        best_prior = max(getattr(v, metric) for v in prior)
        return current_value >= best_prior

    def to_dict(self) -> dict:
        return {"versions": [asdict(mv) for mv in self.versions]}

    def save(self, path: str | Path) -> None:
        """Save the evaluation log to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EvaluationLog:
        """Load an evaluation log from a JSON file.

        Returns an empty log if the file does not exist (first run).
        """
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data = json.load(f)
        versions = [ModelVersion(**v) for v in data.get("versions", [])]
        return cls(versions=versions)


# ── Comparison utilities ──────────────────────────────────────────────────────


def compare_versions(log: EvaluationLog, versions: list[str] | None = None) -> str:
    """Format a comparison table of model versions.

    Args:
        log: EvaluationLog to summarize.
        versions: Version tags to include. Defaults to all recorded versions.

    Returns:
        Formatted string table ready to print or log.
    """
    targets = versions or [mv.version for mv in log.versions]
    records = [log.get(v) for v in targets if log.get(v) is not None]
    if not records:
        return "No versions found in log."

    lines = [
        f"{'Version':<10} {'Type':<15} {'Accuracy':>10} {'Macro F1':>10} {'N_train':>8} {'Dataset'}",
        "-" * 75,
    ]
    for mv in records:
        lines.append(
            f"{mv.version:<10} {mv.model_type:<15} {mv.accuracy:>10.4f} "
            f"{mv.macro_f1:>10.4f} {mv.n_train:>8} {mv.dataset}"
        )
    return "\n".join(lines)


# ── Version-tagged model saving ───────────────────────────────────────────────


def version_model_path(base_dir: str | Path, version: str, model_type: str = "xgboost") -> Path:
    """Return a versioned model file path.

    Args:
        base_dir: Base directory for model artifacts.
        version: Version tag (e.g., "v1").
        model_type: Model type string used as file extension hint.

    Returns:
        Path like base_dir/classifier_v1.json (XGBoost) or base_dir/classifier_v1.pkl (sklearn).
    """
    ext = "json" if model_type == "xgboost" else "pkl"
    return Path(base_dir) / f"classifier_{version}.{ext}"


# ── AcT upgrade evaluation interface ─────────────────────────────────────────


def evaluate_act_vs_xgboost(
    X_test: Any,
    y_test: Any,
    xgboost_model: Any,
    act_model: Any,
    class_names: list[str],
) -> dict[str, Any]:
    """Compare AcT (Action Transformer) against XGBoost on the same test set.

    This is the gate function for the v1→v2 upgrade decision.
    AcT replaces XGBoost only if its accuracy exceeds XGBoost on BoxingVI test set.

    Args:
        X_test: Test feature matrix (n_samples, FEATURE_DIM).
        y_test: True integer labels.
        xgboost_model: Trained XGBClassifier.
        act_model: Trained AcT model (must expose predict(X) → labels).
        class_names: Class name list for evaluation report.

    Returns:
        Dict with "xgboost" and "act" evaluation results and "recommendation".
    """
    from track_b.b2.classifier import evaluate_classifier

    xgb_result = evaluate_classifier(xgboost_model, X_test, y_test, class_names)
    act_result = evaluate_classifier(act_model, X_test, y_test, class_names)

    recommendation = (
        "upgrade_to_act"
        if act_result["accuracy"] > xgboost_model["accuracy"]
        else "keep_xgboost"
    )

    return {
        "xgboost": xgb_result,
        "act": act_result,
        "accuracy_delta": act_result["accuracy"] - xgb_result["accuracy"],
        "recommendation": recommendation,
    }
