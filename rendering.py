"""Unicode labels for OpenCV BGR images."""

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=32)
def load_font(size):
    for path in (str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"),
                 "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("Не найден шрифт с кириллицей: установите Segoe UI или DejaVu Sans")


def draw_labels(frame, labels):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(image)
    for text, position, bgr, size in labels:
        available = max(1, frame.shape[1] - position[0] - 5)
        font = load_font(size)
        while font.getlength(text) > available and size > 8:
            size -= 1
            font = load_font(size)
        painter.text(position, text, font=font, fill=tuple(reversed(bgr)))
    return np.asarray(image)[:, :, ::-1].copy()
