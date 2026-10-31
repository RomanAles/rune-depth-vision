"""Shared OpenCV capture settings for ARDOR GAMING RUNE."""

import argparse
import math
import sys

import cv2


def add_camera_args(parser):
    parser.add_argument("--camera", type=int, default=2,
                        help="Индекс RUNE в OpenCV, по умолчанию 2")
    parser.add_argument("--width", type=int, default=1920,
                        help="Ширина захвата, по умолчанию 1920")
    parser.add_argument("--height", type=int, default=1080,
                        help="Высота захвата, по умолчанию 1080")
    parser.add_argument("--fps", type=float, default=60,
                        help="Запрашиваемая частота камеры, не FPS нейросети")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf"],
                        default="auto", help="Бэкенд камеры Windows")
    parser.add_argument("--fourcc", choices=["MJPG", "YUY2", "native"],
                        default="MJPG", help="Запрашиваемый формат; native = выбор драйвера")
    parser.add_argument("--autofocus", choices=["unchanged", "on", "off"],
                        default="unchanged", help="Автофокус, если управление доступно драйверу")


def validate_camera_args(args):
    if args.camera < 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("Индекс камеры должен быть >= 0, размеры кадра > 0")
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("FPS должен быть конечным положительным числом")
    if args.backend != "auto" and sys.platform != "win32":
        raise ValueError("DirectShow и MSMF доступны только в Windows")


def open_camera(args):
    validate_camera_args(args)
    backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
    if args.backend != "auto":
        candidates = [backends[args.backend]]
    elif sys.platform == "win32":
        candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        candidates = [cv2.CAP_ANY]

    errors = []
    for backend in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(args.camera, backend)
            if not cap.isOpened():
                raise RuntimeError("устройство не открылось")
            settings = []
            if args.fourcc != "native":
                settings.append((cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc)))
            settings.extend([
                (cv2.CAP_PROP_FRAME_WIDTH, args.width),
                (cv2.CAP_PROP_FRAME_HEIGHT, args.height),
                (cv2.CAP_PROP_FPS, args.fps),
            ])
            for prop, value in settings:
                if not cap.set(prop, value):
                    print(f"[WARN] Драйвер не подтвердил настройку камеры {prop}={value}")
            if args.autofocus != "unchanged":
                if not cap.set(cv2.CAP_PROP_AUTOFOCUS, int(args.autofocus == "on")):
                    print("[WARN] Драйвер не подтвердил изменение автофокуса")

            for _ in range(10):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    break
            else:
                raise RuntimeError("не получен первый кадр")

            height, width = frame.shape[:2]
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Камера index={args.camera}, backend={cap.getBackendName()}")
            print(f"[INFO] Запрошено: {args.width}x{args.height} @ {args.fps:g}, {args.fourcc}")
            print(f"[INFO] Кадр: {width}x{height}; FPS по данным драйвера: {reported_fps:g}")
            if (width, height) != (args.width, args.height):
                print("[WARN] Драйвер выбрал другое разрешение; используется фактический размер")
            if not math.isfinite(reported_fps) or abs(reported_fps - args.fps) > 1:
                print("[WARN] Запрошенный FPS не подтверждён драйвером")
            return cap, frame
        except BaseException as exc:
            if cap is not None:
                cap.release()
            if not isinstance(exc, Exception):
                raise
            errors.append(f"backend={backend}: {exc}")

    raise RuntimeError(
        f"Не удалось получить изображение с камеры index={args.camera}. "
        "Проверьте индекс RUNE, USB, разрешения Windows и другие приложения. "
        + "; ".join(errors)
    )


def main():
    parser = argparse.ArgumentParser(description="Проверка захвата ARDOR GAMING RUNE без нейросетей")
    add_camera_args(parser)
    cap, _ = open_camera(parser.parse_args())
    cap.release()


if __name__ == "__main__":
    main()
