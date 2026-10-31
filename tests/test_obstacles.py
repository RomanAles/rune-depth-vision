import unittest

import numpy as np

from gpo.obstacles import analyze_obstacle, StatusFilter


class ObstacleTests(unittest.TestCase):
    def test_partial_missing_depth_is_not_clear(self):
        depth = np.zeros((200, 200))
        depth[110:130, 70:100] = 5
        self.assertEqual(analyze_obstacle(depth, 1.2, 2.5, 0.08, 20)[0], "NO_DEPTH")

    def test_small_connected_obstacle(self):
        depth = np.full((200, 200), 10.0)
        depth[120:126, 90:96] = 0.8
        self.assertEqual(analyze_obstacle(depth, 1.2, 2.5, 0.08, 20)[0], "STOP")

    def test_far_depth_is_valid_even_outside_display_range(self):
        self.assertEqual(analyze_obstacle(np.full((100, 100), 30.0), 1.2, 2.5, 0.08, 20)[0], "CLEAR")

    def test_release_requires_consistent_frames(self):
        state = StatusFilter(3)
        self.assertEqual(state.update("CLEAR"), "CLEAR")
        self.assertEqual(state.update("STOP"), "STOP")
        self.assertEqual(state.update("CLEAR"), "STOP")
        self.assertEqual(state.update("CLEAR"), "STOP")
        self.assertEqual(state.update("CLEAR"), "CLEAR")
        self.assertEqual(state.update("NO_DEPTH"), "NO_DEPTH")
        self.assertEqual(state.update("STOP"), "STOP")

    def test_floor_candidate_mask(self):
        depth = np.full((100, 100), 0.8)
        mask = np.zeros_like(depth, dtype=bool)
        self.assertEqual(analyze_obstacle(depth, 1.2, 2.5, 0.08, 20, candidate_mask=mask)[0], "CLEAR")
