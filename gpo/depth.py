import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA запрошена, но недоступна. Переключаюсь на CPU.")
        return torch.device("cpu")

    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resize_keep_aspect(frame: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_width / w
    target_height = max(1, int(h * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def load_depth_model(model_name: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)

    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()

    return processor, model


@torch.no_grad()
def predict_depth(
    frame_bgr: np.ndarray,
    processor,
    model,
    device: torch.device,
    infer_width: int
) -> np.ndarray:
    original_h, original_w = frame_bgr.shape[:2]

    small_bgr = resize_keep_aspect(frame_bgr, infer_width)
    rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)

    inputs = processor(images=image, return_tensors="pt",
                       size={"height": small_bgr.shape[0], "width": small_bgr.shape[1]})
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if device.type == "cuda":
        inputs = {
            k: v.half() if torch.is_floating_point(v) else v
            for k, v in inputs.items()
        }

    outputs = model(**inputs)
    depth = outputs.predicted_depth

    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    )

    depth = depth.squeeze().float().cpu().numpy()

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0

    return depth
