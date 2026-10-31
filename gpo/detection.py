import numpy as np

from rendering import draw_labels


def object_distances(result, depth=None):
    objects = []
    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].item())
        item = {"class_id": class_id, "label": result.names[class_id],
                "confidence": float(box.conf[0].item()), "xyxy": xyxy, "depth_m": None}
        if depth is not None:
            h, w = depth.shape
            x1, y1, x2, y2 = xyxy
            dx, dy = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            left, top = min(w, max(0, int(x1 + dx))), min(h, max(0, int(y1 + dy)))
            right, bottom = max(0, min(w, int(x2 - dx))), max(0, min(h, int(y2 - dy)))
            region = depth[top:bottom, left:right]
            valid = np.isfinite(region) & (region > 0.1)
            if region.size and valid.sum() >= 10 and valid.mean() >= 0.5:
                item["depth_m"] = float(np.median(region[valid]))
        objects.append(item)
    return objects


def draw_object_distances(frame, objects):
    labels = []
    for item in objects:
        if item["depth_m"] is None:
            continue
        x1, _, _, y2 = item["xyxy"]
        position = (max(0, min(frame.shape[1] - 60, int(x1))),
                    max(0, min(frame.shape[0] - 24, int(y2))))
        labels.append((f"~{item['depth_m']:.2f} м", position, (255, 255, 0), 18))
    return draw_labels(frame, labels) if labels else frame
