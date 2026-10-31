import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpo.geometry import CameraCalibration, height_above_ground, project_pixels


class GeometryTests(unittest.TestCase):
    def calibration(self):
        return CameraCalibration(1920, 1080, 1000, 1100, 960, 540, np.zeros(5))

    def test_intrinsics_scale_with_resolution(self):
        self.assertEqual(self.calibration().intrinsics(960, 540), (500, 550, 480, 270))
        with self.assertRaises(ValueError):
            self.calibration().intrinsics(640, 480)

    def test_optical_axis_and_vertical_projection(self):
        x, y = project_pixels(np.array([960, 1060]), np.array([540, 650]),
                              np.array([2, 2]), 1000, 1100, 960, 540)
        np.testing.assert_allclose(x, [0, 0.2])
        np.testing.assert_allclose(y, [0, 0.2])

    def test_height_with_camera_tilt(self):
        self.assertAlmostEqual(float(height_above_ground(0, 2, 1, 30)), 0)
        self.assertEqual(float(height_above_ground(0.5, 2, 1, 0)), 0.5)

    def test_zero_distortion_preserves_frame(self):
        frame = np.arange(108 * 192 * 3, dtype=np.uint8).reshape(108, 192, 3)
        np.testing.assert_array_equal(self.calibration().rectifier(192, 108)(frame), frame)

    def test_invalid_calibration(self):
        with tempfile.TemporaryDirectory(dir=".venv") as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps({"width": 0}), encoding="utf-8")
            with self.assertRaises(ValueError):
                CameraCalibration.load(path)
