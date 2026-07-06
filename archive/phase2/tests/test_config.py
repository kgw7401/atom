"""Tests for boxing.yaml config loading."""

import pytest

from ml.configs import BoxingConfig, load_boxing_config


@pytest.fixture
def config():
    return BoxingConfig()


class TestConfigLoading:
    def test_load_raw(self):
        raw = load_boxing_config()
        assert raw["domain"] == "boxing"
        assert "actions" in raw
        assert "keypoints" in raw
        assert "pipeline" in raw
        assert "mastery" in raw

    def test_boxing_config_init(self, config):
        assert config is not None


class TestActions:
    def test_action_classes(self, config):
        classes = config.action_classes
        assert len(classes) == 11
        assert "jab" in classes
        assert "cross" in classes
        assert "lead_hook" in classes
        assert "rear_hook" in classes
        assert "lead_uppercut" in classes
        assert "rear_uppercut" in classes
        assert "body_shot" in classes
        assert "slip" in classes
        assert "duck" in classes
        assert "backstep" in classes
        assert "idle" in classes

    def test_num_classes(self, config):
        assert config.num_classes == 11

    def test_action_to_index_roundtrip(self, config):
        a2i = config.action_to_index
        i2a = config.index_to_action
        for action, idx in a2i.items():
            assert i2a[idx] == action

    def test_default_korean_all_actions_covered(self, config):
        korean = config.default_korean
        for action in config.action_classes:
            assert action in korean, f"Missing Korean name for {action}"

    def test_actions_to_korean(self, config):
        result = config.actions_to_korean(["jab", "cross", "lead_hook"])
        assert result == "잽-크로스-훅"

    def test_actions_to_korean_single(self, config):
        assert config.actions_to_korean(["jab"]) == "잽"

    def test_combo_key(self, config):
        key = config.combo_key(["jab", "cross", "lead_hook"])
        assert key == "jab-cross-lead_hook"


class TestKeypoints:
    def test_subset_count(self, config):
        assert config.num_joints == 15

    def test_subset_indices(self, config):
        indices = config.subset_indices
        assert len(indices) == 15
        # Must be valid MediaPipe Pose indices (0-32)
        for idx in indices:
            assert 0 <= idx <= 32

    def test_joint_names_match_indices(self, config):
        assert len(config.joint_names) == len(config.subset_indices)

    def test_center_joints_valid(self, config):
        for j in config.center_joints:
            assert 0 <= j < config.num_joints

    def test_scale_joints_valid(self, config):
        for j in config.scale_joints:
            assert 0 <= j < config.num_joints

    def test_key_joints_present(self, config):
        names = config.joint_names
        for required in ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder",
                         "left_hip", "right_hip", "nose"]:
            assert required in names, f"Missing required joint: {required}"


class TestPipeline:
    def test_fps(self, config):
        assert config.fps == 30

    def test_window_size(self, config):
        assert config.window_size == 30
        # 1 second at 30fps
        assert config.window_size / config.fps == 1.0

    def test_stride(self, config):
        assert config.stride == 5
        assert config.stride < config.window_size

    def test_thresholds_in_range(self, config):
        assert 0.0 < config.action_confidence_threshold <= 1.0
        assert 0.0 < config.pose_visibility_threshold <= 1.0

    def test_combo_gap(self, config):
        assert config.combo_gap_threshold == 0.8

    def test_session_match_window(self, config):
        assert config.session_match_window == 3.0


class TestMastery:
    def test_ema_alpha(self, config):
        assert 0.0 < config.ema_alpha < 1.0
        assert config.ema_alpha == 0.3

    def test_mastery_states_progression(self, config):
        states = config.mastery_states
        assert states["learning"]["min_success_rate"] < states["proficient"]["min_success_rate"]
        assert states["proficient"]["min_success_rate"] < states["mastered"]["min_success_rate"]

    def test_mastery_states_have_required_fields(self, config):
        states = config.mastery_states
        assert "min_success_rate" in states["learning"]
        assert "min_success_rate" in states["proficient"]
        assert "min_sessions" in states["proficient"]
        assert "min_success_rate" in states["mastered"]
        assert "min_consecutive" in states["mastered"]
