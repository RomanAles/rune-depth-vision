import contextlib
import io
import unittest

import numpy as np
import torch
from types import SimpleNamespace
from transformers import DPTImageProcessor

from gpo.depth import predict_depth

from lidar import analyze_obstacle, depth_to_colormap, draw_status, make_pseudo_lidar_view, parse_args


class DepthTests(unittest.TestCase):
    def test_infer_width_changes_network_input(self):
        processor = DPTImageProcessor(ensure_multiple_of=14, keep_aspect_ratio=True)
        shapes = []

        def model(pixel_values):
            shapes.append(pixel_values.shape[-2:])
            return SimpleNamespace(predicted_depth=torch.ones((1, *pixel_values.shape[-2:])))

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for width in (518, 384):
            result = predict_depth(frame, processor, model, torch.device("cpu"), width)
            self.assertEqual(result.shape, (360, 640))
        self.assertGreater(shapes[0][1], shapes[1][1])
        self.assertTrue(all(d % 14 == 0 for shape in shapes for d in shape))

    def test_invalid_arguments(self):
        for argv in (["--stride", "0"], ["--fx", "0"], ["--fov-deg", "180"],
                     ["--max-depth", "nan"], ["--stop-distance", "4"],
                     ["--danger-ratio", "nan"], ["--infer-width", "-1"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)

    def test_statuses(self):
        for value, expected in [(0, "NO_DEPTH"), (float("nan"), "NO_DEPTH"),
                                (0.8, "STOP"), (2, "SLOW"), (5, "CLEAR")]:
            with self.subTest(value=value):
                result = analyze_obstacle(np.full((100, 100), value), 1.2, 2.5, 0.08, 20)
                self.assertEqual(result[0], expected)

    def test_invalid_depth_is_black(self):
        depth = np.array([[0, np.nan, np.inf, -1, 2]], dtype=np.float32)
        colored = depth_to_colormap(depth, 20)
        self.assertFalse(colored[0, :4].any())
        self.assertTrue(colored[0, 4].any())

    def test_custom_threshold_changes_map_color(self):
        depth = np.full((50, 50), 2.2, dtype=np.float32)
        default = make_pseudo_lidar_view(depth, 50, 50, 25, 25, 20, 5, 4)
        custom = make_pseudo_lidar_view(depth, 50, 50, 25, 25, 20, 5, 4, 3, 4)
        self.assertFalse(np.all(default == (0, 0, 255), axis=-1).any())
        self.assertTrue(np.all(custom == (0, 0, 255), axis=-1).any())

    def test_unicode_overlay_does_not_modify_input(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        output = draw_status(frame, "STOP", 0.5, 1, 10)
        self.assertEqual(output.shape, frame.shape)
        self.assertTrue(output[:100].any())
        self.assertFalse(frame.any())


if __name__ == "__main__":
    unittest.main()
