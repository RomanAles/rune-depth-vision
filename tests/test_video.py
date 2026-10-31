import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from gpo.recording import TimingStats
from gpo.runtime import main


class VideoTests(unittest.TestCase):
    def test_headless_video_recording_and_depth_export(self):
        with tempfile.TemporaryDirectory(dir=".venv") as folder:
            root = Path(folder)
            source = root / "source.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
            self.assertTrue(writer.isOpened())
            for color in (30, 80, 120):
                writer.write(np.full((120, 160, 3), color, dtype=np.uint8))
            writer.release()
            with patch("gpo.runtime.select_device", return_value=torch.device("cpu")), \
                 patch("gpo.runtime.load_depth_model", return_value=(None, None)), \
                 patch("gpo.runtime.predict_depth", return_value=np.full((120, 160), 5.0)), \
                 patch("gpo.runtime.cv2.imshow") as imshow, contextlib.redirect_stdout(io.StringIO()):
                summary = main(["--source", str(source), "--headless", "--max-frames", "3",
                                "--output-dir", str(root / "output"), "--save-depth"])
            self.assertEqual(summary["frames"], 3)
            imshow.assert_not_called()
            run = next((root / "output").iterdir())
            events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 3)
            self.assertTrue(all(e["status"] == "CLEAR" for e in events))
            self.assertIn("total", events[-1]["timings_ms"])
            self.assertEqual(len(list(run.glob("depth-*.npy"))), 3)
            cap = cv2.VideoCapture(str(run / "rgb.avi"))
            try:
                self.assertTrue(cap.isOpened())
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                self.assertTrue(cap.read()[0])
            finally:
                cap.release()

    def test_timing_history_is_bounded(self):
        stats = TimingStats()
        for _ in range(5000):
            stats.add({"total": 10})
        self.assertEqual(stats.count, 5000)
        self.assertEqual(len(stats.samples["total"]), 4096)
