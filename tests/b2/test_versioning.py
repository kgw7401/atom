"""Tests for B2 Task 9: Model versioning and evaluation log."""

from pathlib import Path

import pytest

from track_b.b2.versioning import (
    EvaluationLog,
    ModelVersion,
    compare_versions,
    version_model_path,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_metrics(accuracy: float = 0.85) -> dict:
    classes = ["jab", "cross", "lead_hook"]
    per_class = {
        cls: {"precision": accuracy, "recall": accuracy, "f1": accuracy, "support": 20}
        for cls in classes
    }
    return {"accuracy": accuracy, "per_class": per_class}


def make_log_with_versions() -> EvaluationLog:
    log = EvaluationLog()
    log.record("v1", "xgboost", make_metrics(0.82), n_train=5000, n_test=1000, dataset="boxingvi")
    log.record("v1.1", "xgboost", make_metrics(0.87), n_train=6000, n_test=1000, dataset="boxingvi+olympic")
    log.record("v2", "random_forest", make_metrics(0.75), n_train=6000, n_test=1000, dataset="boxingvi")
    return log


# ── ModelVersion ──────────────────────────────────────────────────────────────

class TestModelVersion:
    def test_macro_f1_calculated(self):
        mv = ModelVersion(
            version="v1", model_type="xgboost", accuracy=0.85,
            per_class={"jab": {"f1": 0.9}, "cross": {"f1": 0.8}},
            n_train=100, n_test=30, dataset="boxingvi",
        )
        assert mv.macro_f1 == pytest.approx(0.85)

    def test_macro_f1_empty_per_class(self):
        mv = ModelVersion(
            version="v1", model_type="xgboost", accuracy=0.0,
            per_class={}, n_train=0, n_test=0, dataset="boxingvi",
        )
        assert mv.macro_f1 == pytest.approx(0.0)

    def test_timestamp_auto_set(self):
        mv = ModelVersion(
            version="v1", model_type="xgboost", accuracy=0.85,
            per_class={}, n_train=100, n_test=30, dataset="boxingvi",
        )
        assert mv.timestamp != ""
        assert "T" in mv.timestamp  # ISO format

    def test_notes_default_empty(self):
        mv = ModelVersion(
            version="v1", model_type="xgboost", accuracy=0.85,
            per_class={}, n_train=100, n_test=30, dataset="boxingvi",
        )
        assert mv.notes == ""


# ── EvaluationLog.record ──────────────────────────────────────────────────────

class TestEvaluationLogRecord:
    def test_record_adds_version(self):
        log = EvaluationLog()
        log.record("v1", "xgboost", make_metrics(0.85), n_train=100, n_test=30, dataset="boxingvi")
        assert len(log.versions) == 1

    def test_record_returns_model_version(self):
        log = EvaluationLog()
        mv = log.record("v1", "xgboost", make_metrics(0.85), n_train=100, n_test=30, dataset="boxingvi")
        assert isinstance(mv, ModelVersion)

    def test_record_accuracy_stored(self):
        log = EvaluationLog()
        log.record("v1", "xgboost", make_metrics(0.91), n_train=100, n_test=30, dataset="boxingvi")
        assert log.versions[0].accuracy == pytest.approx(0.91)

    def test_record_multiple_versions(self):
        log = make_log_with_versions()
        assert len(log.versions) == 3

    def test_record_notes_stored(self):
        log = EvaluationLog()
        log.record("v1", "xgboost", make_metrics(), n_train=100, n_test=30,
                   dataset="boxingvi", notes="First run")
        assert log.versions[0].notes == "First run"

    def test_record_model_path_stored(self):
        log = EvaluationLog()
        log.record("v1", "xgboost", make_metrics(), n_train=100, n_test=30,
                   dataset="boxingvi", model_path="models/v1.json")
        assert log.versions[0].model_path == "models/v1.json"


# ── EvaluationLog.get / best_version ─────────────────────────────────────────

class TestEvaluationLogQuery:
    def test_get_existing_version(self):
        log = make_log_with_versions()
        mv = log.get("v1")
        assert mv is not None
        assert mv.version == "v1"

    def test_get_nonexistent_returns_none(self):
        log = make_log_with_versions()
        assert log.get("v99") is None

    def test_best_version_accuracy(self):
        log = make_log_with_versions()
        best = log.best_version("accuracy")
        assert best is not None
        assert best.version == "v1.1"  # 0.87 is highest

    def test_best_version_empty_log(self):
        log = EvaluationLog()
        assert log.best_version() is None

    def test_best_version_macro_f1(self):
        log = make_log_with_versions()
        best = log.best_version("macro_f1")
        assert best is not None
        assert best.accuracy == pytest.approx(0.87)


# ── EvaluationLog.is_improvement ─────────────────────────────────────────────

class TestIsImprovement:
    def test_first_version_is_improvement(self):
        log = EvaluationLog()
        log.record("v1", "xgboost", make_metrics(0.80), n_train=100, n_test=30, dataset="boxingvi")
        assert log.is_improvement("v1")

    def test_higher_accuracy_is_improvement(self):
        log = make_log_with_versions()
        assert log.is_improvement("v1.1")  # 0.87 > 0.82

    def test_lower_accuracy_not_improvement(self):
        log = make_log_with_versions()
        assert not log.is_improvement("v2")  # 0.75 < 0.87

    def test_nonexistent_version_not_improvement(self):
        log = make_log_with_versions()
        assert not log.is_improvement("v99")


# ── EvaluationLog persistence ─────────────────────────────────────────────────

class TestEvaluationLogPersistence:
    def test_save_creates_file(self, tmp_path):
        log = make_log_with_versions()
        path = tmp_path / "eval_log.json"
        log.save(path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        log = make_log_with_versions()
        path = tmp_path / "deep" / "nested" / "eval_log.json"
        log.save(path)
        assert path.exists()

    def test_roundtrip_version_count(self, tmp_path):
        log = make_log_with_versions()
        path = tmp_path / "log.json"
        log.save(path)
        loaded = EvaluationLog.load(path)
        assert len(loaded.versions) == len(log.versions)

    def test_roundtrip_accuracy(self, tmp_path):
        log = make_log_with_versions()
        path = tmp_path / "log.json"
        log.save(path)
        loaded = EvaluationLog.load(path)
        assert loaded.versions[0].accuracy == pytest.approx(log.versions[0].accuracy)

    def test_roundtrip_version_tags(self, tmp_path):
        log = make_log_with_versions()
        path = tmp_path / "log.json"
        log.save(path)
        loaded = EvaluationLog.load(path)
        assert [v.version for v in loaded.versions] == [v.version for v in log.versions]

    def test_load_returns_empty_if_no_file(self, tmp_path):
        log = EvaluationLog.load(tmp_path / "nonexistent.json")
        assert log.versions == []

    def test_accepts_string_path(self, tmp_path):
        log = make_log_with_versions()
        log.save(str(tmp_path / "log.json"))
        loaded = EvaluationLog.load(str(tmp_path / "log.json"))
        assert len(loaded.versions) == 3


# ── compare_versions ──────────────────────────────────────────────────────────

class TestCompareVersions:
    def test_returns_string(self):
        log = make_log_with_versions()
        result = compare_versions(log)
        assert isinstance(result, str)

    def test_contains_version_tags(self):
        log = make_log_with_versions()
        result = compare_versions(log)
        assert "v1" in result
        assert "v1.1" in result
        assert "v2" in result

    def test_contains_accuracy_values(self):
        log = make_log_with_versions()
        result = compare_versions(log)
        assert "0.82" in result or "0.8200" in result

    def test_subset_of_versions(self):
        log = make_log_with_versions()
        result = compare_versions(log, versions=["v1", "v1.1"])
        assert "v1" in result
        assert "v2" not in result

    def test_empty_log_message(self):
        log = EvaluationLog()
        result = compare_versions(log)
        assert "No versions" in result

    def test_nonexistent_version_skipped(self):
        log = make_log_with_versions()
        result = compare_versions(log, versions=["v1", "v99"])
        assert "v1" in result
        assert "v99" not in result


# ── version_model_path ────────────────────────────────────────────────────────

class TestVersionModelPath:
    def test_xgboost_uses_json_extension(self, tmp_path):
        path = version_model_path(tmp_path, "v1", "xgboost")
        assert path.suffix == ".json"

    def test_sklearn_uses_pkl_extension(self, tmp_path):
        path = version_model_path(tmp_path, "v1", "random_forest")
        assert path.suffix == ".pkl"

    def test_path_contains_version(self, tmp_path):
        path = version_model_path(tmp_path, "v1.2", "xgboost")
        assert "v1.2" in path.name

    def test_returns_path_object(self, tmp_path):
        path = version_model_path(tmp_path, "v1", "xgboost")
        assert isinstance(path, Path)
