from typing import Tuple
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Калибровка должна быть JSON-объектом")
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Неверный параметр калибровки: {key}")
            if key in ("width", "height") and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} должен быть целым > 0")
            if key in ("fx", "fy") and value <= 0:
                raise ValueError(f"{key} должен быть > 0")
        distortion = np.asarray(data.get("distortion", [0, 0, 0, 0, 0]), dtype=np.float64)
        if distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("Некорректные коэффициенты дисторсии")
        return cls(**{k: data[k] for k in ("width", "height", "fx", "fy", "cx", "cy")},
                   distortion=distortion)

    def intrinsics(self, width, height):
        if not math.isclose(width / height, self.width / self.height, rel_tol=0.01):
            raise ValueError("Соотношение сторон кадра отличается от калибровки; нужна калибровка этого режима")
        sx, sy = width / self.width, height / self.height
        return self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy

    def rectifier(self, width, height):
        fx, fy, cx, cy = self.intrinsics(width, height)
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        maps = cv2.initUndistortRectifyMap(matrix, self.distortion, None, matrix,
                                         (width, height), cv2.CV_32FC1)
        return lambda frame: cv2.remap(frame, *maps, interpolation=cv2.INTER_LINEAR)


def project_pixels(u, v, z, fx, fy, cx, cy):
    return (u - cx) * z / fx, (v - cy) * z / fy


def height_above_ground(y, z, camera_height, pitch_deg):
    pitch = np.deg2rad(pitch_deg)
    return camera_height - (y * np.cos(pitch) + z * np.sin(pitch))


def height_mask(depth, fy, cy, camera_height, pitch_deg, min_height, max_height):
    rows = np.arange(depth.shape[0], dtype=np.float32)[:, None]
    y = (rows - cy) * depth / fy
    heights = height_above_ground(y, depth, camera_height, pitch_deg)
    return np.isfinite(heights) & (heights >= min_height) & (heights <= max_height)


def estimate_intrinsics(
    width: int,
    height: int,
    fov_deg: float,
    fx=None,
    fy=None,
    cx=None,
    cy=None
) -> Tuple[float, float, float, float]:
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    if fx is None:
        fov_rad = np.deg2rad(fov_deg)
        fx = width / (2.0 * np.tan(fov_rad / 2.0))

    if fy is None:
        fy = fx

    return float(fx), float(fy), float(cx), float(cy)
