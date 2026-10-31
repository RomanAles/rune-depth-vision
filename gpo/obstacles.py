import numpy as np
import cv2


def roi_bounds(shape, roi):
    h, w = shape[:2]
    x1, y1, x2, y2 = roi
    return int(w * x1), int(h * y1), int(w * x2), int(h * y2)


def has_component(mask, min_pixels):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return count > 1 and bool((stats[1:, cv2.CC_STAT_AREA] >= min_pixels).any())


class StatusFilter:
    def __init__(self, release_frames=3):
        self.release_frames = release_frames
        self.status = None
        self.pending = None
        self.count = 0

    def update(self, status):
        severity = {"CLEAR": 0, "SLOW": 1, "STOP": 2, "NO_DEPTH": 3}
        if self.status is None or status in ("STOP", "NO_DEPTH") or severity[status] >= severity[self.status]:
            self.status, self.pending, self.count = status, None, 0
        else:
            self.count = self.count + 1 if self.pending == status else 1
            self.pending = status
            if self.count >= self.release_frames:
                self.status, self.pending, self.count = status, None, 0
        return self.status


def analyze_obstacle(
    depth: np.ndarray,
    stop_distance: float,
    slow_distance: float,
    danger_ratio: float,
    max_depth: float,
    roi=(0.35, 0.55, 0.65, 0.95),
    min_valid_ratio=0.5,
    min_component_pixels=25,
    candidate_mask=None
):
    x1, y1, x2, y2 = roi_bounds(depth.shape, roi)
    region = depth[y1:y2, x1:x2]
    valid = np.isfinite(region) & (region > 0.1)
    if region.size == 0 or valid.sum() < 100 or valid.mean() < min_valid_ratio:
        return "NO_DEPTH", 0.0, None
    eligible = valid.copy()
    if candidate_mask is not None:
        eligible &= candidate_mask[y1:y2, x1:x2]
    stop_pixels = eligible & (region < stop_distance)
    slow_pixels = eligible & (region < slow_distance)
    stop_ratio = float(stop_pixels.sum() / valid.sum())
    slow_ratio = float(slow_pixels.sum() / valid.sum())
    valid_depth = region[valid]

    median_depth = float(np.median(valid_depth))

    if stop_ratio > danger_ratio or has_component(stop_pixels, min_component_pixels):
        return "STOP", stop_ratio, median_depth

    if slow_ratio > danger_ratio or has_component(slow_pixels, min_component_pixels):
        return "SLOW", slow_ratio, median_depth

    return "CLEAR", slow_ratio, median_depth
