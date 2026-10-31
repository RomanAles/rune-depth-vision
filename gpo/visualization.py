import cv2
import numpy as np
from rendering import draw_labels
from .geometry import project_pixels, height_above_ground
from .obstacles import roi_bounds


def depth_to_colormap(depth: np.ndarray, max_depth: float) -> np.ndarray:
    depth_clipped = np.clip(np.nan_to_num(depth, nan=0, posinf=0, neginf=0), 0, max_depth)

    normalized = depth_clipped / max_depth
    normalized = 1.0 - normalized

    depth_u8 = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    colored[~np.isfinite(depth) | (depth <= 0.1)] = 0

    return colored


def make_pseudo_lidar_view(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float,
    side_range: float,
    stride: int,
    stop_distance: float = 1.2,
    slow_distance: float = 2.5,
    camera_height=None,
    pitch_deg: float = 0,
    min_height: float = 0.05,
    max_height: float = 2.0
) -> np.ndarray:
    h, w = depth.shape[:2]

    scale = 35
    bev_w = int(side_range * 2 * scale)
    bev_h = int(max_depth * scale)

    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    center_x = bev_w // 2

    y_start = int(h * 0.40)

    vv = np.arange(y_start, h, stride)
    uu = np.arange(0, w, stride)

    if vv.size > 0 and uu.size > 0:
        grid_v, grid_u = np.meshgrid(vv, uu, indexing="ij")
        z = depth[grid_v, grid_u].astype(np.float32)

        # Back-project sampled pixels into camera coordinates.
        x, y = project_pixels(grid_u, grid_v, z, fx, fy, cx, cy)

        pitch = np.deg2rad(pitch_deg)
        forward = z * np.cos(pitch) - y * np.sin(pitch)
        valid = np.isfinite(z) & (z > 0.1) & (forward > 0.1) & (forward <= max_depth)
        valid &= np.abs(x) <= side_range
        if camera_height is None:
            valid &= y >= -0.5
        else:
            heights = height_above_ground(y, z, camera_height, pitch_deg)
            valid &= (heights >= min_height) & (heights <= max_height)

        if np.any(valid):
            z_valid = forward[valid]
            axial_depth = z[valid]
            x_valid = x[valid]

            px = np.clip((center_x + x_valid * scale).astype(np.int32), 0, bev_w - 1)
            py = np.clip((bev_h - z_valid * scale).astype(np.int32), 0, bev_h - 1)

            close_mask = axial_depth < stop_distance
            mid_mask = (axial_depth >= stop_distance) & (axial_depth < slow_distance)
            far_mask = axial_depth >= slow_distance

            bev[py[far_mask], px[far_mask]] = (0, 255, 0)
            bev[py[mid_mask], px[mid_mask]] = (0, 255, 255)
            bev[py[close_mask], px[close_mask]] = (0, 0, 255)

    cv2.line(bev, (center_x, bev_h), (center_x, 0), (255, 255, 255), 1)

    for meter in range(1, int(max_depth) + 1):
        y = int(bev_h - meter * scale)
        if 0 <= y < bev_h:
            cv2.line(bev, (0, y), (bev_w, y), (40, 40, 40), 1)
            cv2.putText(
                bev,
                f"{meter} m",
                (5, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (160, 160, 160),
                1,
                cv2.LINE_AA
            )

    return draw_labels(bev, [("Pseudo-LiDAR: вид сверху", (10, 5), (255, 255, 255), 18)])


def draw_status(
    frame: np.ndarray,
    status: str,
    ratio: float,
    median_depth,
    fps: float,
    roi=(0.35, 0.55, 0.65, 0.95)
) -> np.ndarray:
    output = frame.copy()

    if status == "STOP":
        color = (0, 0, 255)
        text = "STOP: препятствие впереди"
    elif status == "SLOW":
        color = (0, 255, 255)
        text = "SLOW: объект рядом"
    elif status == "CLEAR":
        color = (0, 255, 0)
        text = "CLEAR"
    else:
        color = (255, 255, 255)
        text = "НЕТ НАДЕЖНОЙ ГЛУБИНЫ"

    cv2.rectangle(output, (5, 5), (min(output.shape[1] - 1, 620), 110), (0, 0, 0), -1)

    if median_depth is not None:
        depth_text = f"медианная глубина: {median_depth:.2f} м | риск: {ratio:.3f}"
    else:
        depth_text = f"риск: {ratio:.3f}"

    output = draw_labels(output, [
        (text, (15, 10), color, 26),
        (depth_text, (15, 48), (230, 230, 230), 18),
        (f"FPS: {fps:.1f}", (15, 78), (230, 230, 230), 18),
    ])

    x1, y1, x2, y2 = roi_bounds(output.shape, roi)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    return output
