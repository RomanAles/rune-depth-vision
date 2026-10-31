import argparse
import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from camera import add_camera_args, open_camera


class CameraTests(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        add_camera_args(parser)
        self.args = parser.parse_args([])
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def capture(self, width=1920, height=1080):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.set.return_value = True
        cap.read.return_value = (True, np.zeros((height, width, 3), dtype=np.uint8))
        cap.get.return_value = 60.0
        cap.getBackendName.return_value = "DSHOW"
        return cap

    @patch("camera.sys.platform", "win32")
    def test_rune_settings_and_first_frame(self):
        cap = self.capture()
        with patch("camera.cv2.VideoCapture", return_value=cap) as factory:
            result, frame = open_camera(self.args)
        factory.assert_called_once_with(2, cv2.CAP_DSHOW)
        self.assertIs(result, cap)
        self.assertEqual(frame.shape[:2], (1080, 1920))
        cap.set.assert_any_call(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set.assert_any_call(cv2.CAP_PROP_FPS, 60)
        self.assertFalse(any(c.args[0] == cv2.CAP_PROP_AUTOFOCUS for c in cap.set.call_args_list))
        cap.release.assert_not_called()

    @patch("camera.sys.platform", "win32")
    def test_fallback_when_first_backend_has_no_frames(self):
        failed, working = self.capture(), self.capture()
        failed.read.return_value = (False, None)
        with patch("camera.cv2.VideoCapture", side_effect=[failed, working]) as factory:
            result, _ = open_camera(self.args)
        self.assertIs(result, working)
        failed.release.assert_called_once()
        self.assertEqual(factory.call_args_list[-1].args, (2, cv2.CAP_MSMF))

    @patch("camera.sys.platform", "win32")
    def test_actual_frame_size_and_rejected_settings(self):
        cap = self.capture(640, 480)
        cap.set.return_value = False
        with patch("camera.cv2.VideoCapture", return_value=cap):
            _, frame = open_camera(self.args)
        self.assertEqual(frame.shape[:2], (480, 640))

    @patch("camera.sys.platform", "win32")
    def test_native_format_and_focus_off(self):
        self.args.fourcc = "native"
        self.args.autofocus = "off"
        cap = self.capture()
        with patch("camera.cv2.VideoCapture", return_value=cap):
            open_camera(self.args)
        cap.set.assert_any_call(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.assertFalse(any(c.args[0] == cv2.CAP_PROP_FOURCC for c in cap.set.call_args_list))

    @patch("camera.sys.platform", "win32")
    def test_all_failures_release_devices(self):
        caps = [self.capture(), self.capture()]
        for cap in caps:
            cap.isOpened.return_value = False
        with patch("camera.cv2.VideoCapture", side_effect=caps):
            with self.assertRaisesRegex(RuntimeError, "index=2"):
                open_camera(self.args)
        for cap in caps:
            cap.release.assert_called_once()

    def test_invalid_parameters_do_not_open_device(self):
        for key, value in [("camera", -1), ("width", 0), ("height", -1),
                           ("fps", 0), ("fps", float("nan"))]:
            original = getattr(self.args, key)
            with self.subTest(key=key, value=value):
                setattr(self.args, key, value)
                with patch("camera.cv2.VideoCapture") as factory:
                    with self.assertRaises(ValueError):
                        open_camera(self.args)
                    factory.assert_not_called()
                setattr(self.args, key, original)


if __name__ == "__main__":
    unittest.main()
