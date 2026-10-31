import json
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2
import numpy as np


class TimingStats:
    def __init__(self):
        self.samples = defaultdict(lambda: deque(maxlen=4096))
        self.count = 0
        self.started = time.perf_counter()

    def add(self, timings):
        self.count += 1
        for key, value in timings.items():
            self.samples[key].append(value)

    def summary(self):
        return {"frames": self.count, "elapsed_s": time.perf_counter() - self.started,
                "percentile_window": 4096,
                "timings_ms": {key: {"p50": float(np.percentile(values, 50)),
                                      "p95": float(np.percentile(values, 95)), "samples": len(values)}
                               for key, values in self.samples.items() if values}}


class RunRecorder:
    def __init__(self, directory, args):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"run-{stamp}-{uuid.uuid4().hex[:6]}"
        self.path.mkdir(parents=True)
        self.fps = args.output_fps
        self.save_depth = args.save_depth
        self.writers = {}
        self.events = (self.path / "events.jsonl").open("w", encoding="utf-8")
        try:
            (self.path / "settings.json").write_text(
                json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8")
        except BaseException:
            self.events.close()
            raise

    def write(self, index, views, depth):
        for name, frame in views.items():
            if name not in self.writers:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.path / f"{name}.avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Не удалось создать видео {name}")
                self.writers[name] = writer
            self.writers[name].write(frame)
        if self.save_depth and depth is not None:
            np.save(self.path / f"depth-{index:06d}.npy", depth)
    def write_event(self, event):
        self.events.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.events.flush()

    def close(self, summary):
        for writer in self.writers.values():
            writer.release()
        self.events.close()
        (self.path / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
