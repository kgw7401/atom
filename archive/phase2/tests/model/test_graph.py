"""Tests for boxing skeleton graph topology."""

import numpy as np
import pytest

from ml.graph.boxing_graph import (
    JOINT_NAMES,
    MEDIAPIPE_TO_SUBSET,
    SUBSET_INDICES,
    center,
    get_adjacency_matrix,
    get_edge_sets,
    inward_edges,
    num_node,
    outward_edges,
    self_link,
)


class TestGraphConstants:
    def test_num_nodes(self):
        assert num_node == 15

    def test_joint_names_count(self):
        assert len(JOINT_NAMES) == num_node

    def test_subset_indices_count(self):
        assert len(SUBSET_INDICES) == num_node

    def test_mediapipe_mapping_count(self):
        assert len(MEDIAPIPE_TO_SUBSET) == num_node

    def test_mediapipe_indices_valid(self):
        for mp_idx in SUBSET_INDICES:
            assert 0 <= mp_idx <= 32, f"Invalid MediaPipe index: {mp_idx}"

    def test_subset_indices_match_mapping(self):
        assert SUBSET_INDICES == list(MEDIAPIPE_TO_SUBSET.keys())

    def test_center_is_valid_node(self):
        assert 0 <= center < num_node


class TestEdgeSets:
    def test_self_link_count(self):
        assert len(self_link) == num_node

    def test_self_link_diagonal(self):
        for i, j in self_link:
            assert i == j

    def test_inward_outward_same_count(self):
        assert len(inward_edges) == len(outward_edges)

    def test_outward_is_reverse_of_inward(self):
        reversed_inward = [(j, i) for (i, j) in inward_edges]
        assert set(outward_edges) == set(reversed_inward)

    def test_edge_indices_valid(self):
        for edges in [inward_edges, outward_edges]:
            for i, j in edges:
                assert 0 <= i < num_node, f"Invalid edge source: {i}"
                assert 0 <= j < num_node, f"Invalid edge target: {j}"

    def test_no_duplicate_edges(self):
        assert len(set(inward_edges)) == len(inward_edges)
        assert len(set(outward_edges)) == len(outward_edges)

    def test_get_edge_sets_returns_three(self):
        sl, inw, outw = get_edge_sets()
        assert len(sl) == num_node
        assert len(inw) == len(inward_edges)
        assert len(outw) == len(outward_edges)

    def test_graph_is_connected(self):
        """Every node should be reachable from every other node (undirected)."""
        all_edges = inward_edges + outward_edges
        adj = {i: set() for i in range(num_node)}
        for i, j in all_edges:
            adj[i].add(j)
            adj[j].add(i)

        # BFS from node 0
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)

        assert visited == set(range(num_node)), \
            f"Disconnected nodes: {set(range(num_node)) - visited}"


class TestAdjacencyMatrix:
    @pytest.fixture
    def A(self):
        return get_adjacency_matrix()

    def test_shape(self, A):
        assert A.shape == (3, num_node, num_node)

    def test_dtype(self, A):
        assert A.dtype == np.float32

    def test_identity_subset_is_diagonal(self, A):
        """A[0] (self-link) should be identity matrix."""
        expected = np.eye(num_node, dtype=np.float32)
        np.testing.assert_array_equal(A[0], expected)

    def test_no_negative_values(self, A):
        assert np.all(A >= 0)

    def test_row_normalized(self, A):
        """Each row of each subset should sum to ≤ 1.0 (normalized)."""
        for k in range(3):
            row_sums = A[k].sum(axis=1)
            for i in range(num_node):
                if row_sums[i] > 0:
                    assert abs(row_sums[i] - 1.0) < 1e-5, \
                        f"A[{k}] row {i} sums to {row_sums[i]}"

    def test_inward_has_expected_edges(self, A):
        """A[1] should have non-zero entries for inward edges."""
        for src, dst in inward_edges:
            assert A[1][src, dst] > 0, \
                f"Missing inward edge: {src} → {dst}"

    def test_outward_has_expected_edges(self, A):
        """A[2] should have non-zero entries for outward edges."""
        for src, dst in outward_edges:
            assert A[2][src, dst] > 0, \
                f"Missing outward edge: {src} → {dst}"

    def test_inward_no_self_loops(self, A):
        """A[1] diagonal should be all zeros."""
        for i in range(num_node):
            assert A[1][i, i] == 0

    def test_outward_no_self_loops(self, A):
        """A[2] diagonal should be all zeros."""
        for i in range(num_node):
            assert A[2][i, i] == 0


class TestSkeletonStructure:
    """Verify the graph represents a valid boxing skeleton."""

    def test_wrists_connect_to_elbows(self):
        # left_wrist(7) → left_elbow(5)
        assert (7, 5) in inward_edges
        # right_wrist(8) → right_elbow(6)
        assert (8, 6) in inward_edges

    def test_elbows_connect_to_shoulders(self):
        assert (5, 3) in inward_edges  # left_elbow → left_shoulder
        assert (6, 4) in inward_edges  # right_elbow → right_shoulder

    def test_shoulders_connect_to_hips(self):
        assert (3, 9) in inward_edges   # left_shoulder → left_hip
        assert (4, 10) in inward_edges  # right_shoulder → right_hip

    def test_hips_connected(self):
        assert (10, 9) in inward_edges  # right_hip → left_hip

    def test_leg_chain(self):
        assert (13, 11) in inward_edges  # left_ankle → left_knee
        assert (11, 9) in inward_edges   # left_knee → left_hip
        assert (14, 12) in inward_edges  # right_ankle → right_knee
        assert (12, 10) in inward_edges  # right_knee → right_hip

    def test_head_connections(self):
        assert (1, 0) in inward_edges  # left_ear → nose
        assert (2, 0) in inward_edges  # right_ear → nose
        assert (3, 0) in inward_edges  # left_shoulder → nose
        assert (4, 0) in inward_edges  # right_shoulder → nose
