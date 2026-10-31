import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from gpo.config import parse_args
from gpo.runtime import main


class ConfigTests(unittest.TestCase):
    def test_json_and_cli_precedence(self):
        with tempfile.TemporaryDirectory(dir=".venv") as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"camera": 5, "mode": "combined", "fx": None}), encoding="utf-8")
            args = parse_args(["--config", str(path), "--camera", "2"])
        self.assertEqual(args.camera, 2)
        self.assertEqual(args.mode, "combined")

    def test_rejects_unknown_keys_and_wrong_types(self):
        for data in ({"camera": True}, {"camera": "2"}, {"unknown": 1}, {"mode": "invalid"}):
            with self.subTest(data=data), tempfile.TemporaryDirectory(dir=".venv") as folder:
                path = Path(folder) / "settings.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(["--config", str(path)])


class RuntimeTests(unittest.TestCase):
    def test_camera_released_when_model_loading_fails(self):
        cap = MagicMock()
        with patch("gpo.runtime.open_camera", return_value=(cap, np.zeros((100, 100, 3), np.uint8))), \
             patch("gpo.runtime.select_device", return_value=torch.device("cpu")), \
             patch("gpo.runtime.load_depth_model", side_effect=RuntimeError("offline")), \
             patch("gpo.runtime.cv2.destroyAllWindows"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                main([])
        cap.release.assert_called_once()

    def test_combined_uses_same_frame_and_handles_interrupt(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        cap, detector = MagicMock(), MagicMock()
        result = MagicMock()
        result.plot.return_value = frame.copy()
        detector.predict.return_value = [result]
        yolo = SimpleNamespace(YOLO=MagicMock(return_value=detector))
        with patch.dict(sys.modules, {"ultralytics": yolo}), \
             patch("gpo.runtime.open_camera", return_value=(cap, frame)), \
             patch("gpo.runtime.select_device", return_value=torch.device("cpu")), \
             patch("gpo.runtime.load_depth_model", return_value=(None, None)), \
             patch("gpo.runtime.predict_depth", return_value=np.full((100, 100), 5.0)) as predict, \
             patch("gpo.runtime.cv2.imshow"), \
             patch("gpo.runtime.cv2.waitKey", side_effect=KeyboardInterrupt()), \
             patch("gpo.runtime.cv2.destroyAllWindows"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                main(["--mode", "combined"])
        self.assertIs(detector.predict.call_args.args[0], frame)
        self.assertIs(predict.call_args.kwargs["frame_bgr"], frame)
        cap.release.assert_called_once()
