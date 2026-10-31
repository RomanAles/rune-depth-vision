import time
import sys
from datetime import datetime, timezone

import cv2

from camera import open_camera
from .config import parse_args
from .depth import select_device, load_depth_model, predict_depth
from .detection import object_distances, draw_object_distances
from .geometry import CameraCalibration, estimate_intrinsics, height_mask
from .obstacles import analyze_obstacle, StatusFilter
from .recording import RunRecorder, TimingStats
from .visualization import draw_status, depth_to_colormap, make_pseudo_lidar_view


def cli(argv=None, default_mode="depth"):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        return main(argv, default_mode)
    except KeyboardInterrupt:
        print("[INFO] Остановлено пользователем")
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def open_source(args):
    if args.source is None:
        return open_camera(args)
    if not args.source.is_file():
        raise ValueError(f"Видеофайл не найден: {args.source}")
    cap = cv2.VideoCapture(str(args.source))
    try:
        ok, frame = cap.read()
        if not cap.isOpened() or not ok or frame is None or not frame.size:
            raise RuntimeError(f"Не удалось прочитать видео: {args.source}")
        return cap, frame
    except BaseException:
        cap.release()
        raise


def main(argv=None, default_mode="depth"):
    args = parse_args(argv, default_mode=default_mode)
    device = select_device(args.device)
    print(f"[INFO] Устройство: {device}; режим: {args.mode}")
    cap, frame = open_source(args)
    recorder = None
    stats = TimingStats()
    frames = 0
    try:
        real_height, real_width = frame.shape[:2]
        intrinsics = estimate_intrinsics(real_width, real_height, args.fov_deg,
                                         args.fx, args.fy, args.cx, args.cy)
        rectify = None
        if args.calibration:
            calibration = CameraCalibration.load(args.calibration)
            intrinsics = calibration.intrinsics(real_width, real_height)
            rectify = calibration.rectifier(real_width, real_height)
        fx, fy, cx, cy = intrinsics
        use_depth = args.mode in ("depth", "combined")
        if use_depth and args.fx is None and not args.calibration:
            print("[WARN] Проекция приближённая: горизонтальный FOV RUNE не откалиброван. "
                  "Паспортные 78 градусов не используются как горизонтальный FOV.")
        if use_depth and args.camera_height is None:
            print("[INFO] Высота камеры не задана: фильтрация относительно пола выключена")
        processor = model = detector = None
        if use_depth:
            print(f"[INFO] Загружаю модель: {args.model}")
            try:
                processor, model = load_depth_model(args.model, device)
            except Exception as exc:
                raise RuntimeError(
                    "Не удалось загрузить модель глубины. Проверьте имя модели, интернет и локальный кэш."
                ) from exc
        if args.mode in ("detect", "combined"):
            from ultralytics import YOLO
            detector = YOLO(str(args.weights))
        if args.output_dir:
            recorder = RunRecorder(args.output_dir, args)
            print(f"[INFO] Запись: {recorder.path}")
        status_filter = StatusFilter(args.release_frames)
        stats.started = time.perf_counter()
        capture_ms = 0.0
        while True:
            started = time.perf_counter()
            timings = {"capture": capture_ms}
            if frame.shape[:2] != (real_height, real_width):
                raise RuntimeError("Размер кадра изменился во время работы; перезапустите источник")
            if rectify:
                frame = rectify(frame)
            depth = result = None
            raw_status = status = None
            ratio, median = 0.0, None
            depth_started = time.perf_counter()
            if use_depth:
                depth = predict_depth(frame_bgr=frame, processor=processor, model=model,
                                      device=device, infer_width=args.infer_width)
            timings["depth"] = (time.perf_counter() - depth_started) * 1000
            detect_started = time.perf_counter()
            if detector is not None:
                result = detector.predict(frame, imgsz=args.imgsz, conf=args.confidence,
                                          device=0 if device.type == "cuda" else "cpu", verbose=False)[0]
            timings["detect"] = (time.perf_counter() - detect_started) * 1000
            analyze_started = time.perf_counter()
            objects = object_distances(result, depth) if result is not None else []
            if use_depth:
                candidates = None
                if args.camera_height is not None:
                    candidates = height_mask(depth, fy, cy, args.camera_height, args.pitch_deg,
                                             args.min_height, args.max_height)
                raw_status, ratio, median = analyze_obstacle(
                    depth, args.stop_distance, args.slow_distance, args.danger_ratio,
                    args.max_depth, roi=args.roi, min_valid_ratio=args.min_valid_ratio,
                    min_component_pixels=args.min_component_pixels, candidate_mask=candidates)
                status = status_filter.update(raw_status)
            timings["analyze"] = (time.perf_counter() - analyze_started) * 1000
            render_started = time.perf_counter()
            views = {}
            if not args.headless or recorder is not None:
                annotated = result.plot() if result is not None else frame.copy()
                if use_depth:
                    annotated = draw_object_distances(annotated, objects)
                    fps = 1000 / max(0.01, sum(timings.values()))
                    annotated = draw_status(annotated, status, ratio, median, fps, roi=args.roi)
                    views["depth"] = depth_to_colormap(depth, args.max_depth)
                    views["bev"] = make_pseudo_lidar_view(
                        depth, fx, fy, cx, cy, args.max_depth, args.side_range, args.stride,
                        args.stop_distance, args.slow_distance, args.camera_height,
                        args.pitch_deg, args.min_height, args.max_height)
                views["rgb"] = annotated
            timings["render"] = (time.perf_counter() - render_started) * 1000
            frames += 1
            event = {"frame": frames, "timestamp": datetime.now(timezone.utc).isoformat(),
                     "status": status, "raw_status": raw_status, "risk_ratio": ratio,
                     "median_depth_m": median, "objects": objects, "timings_ms": timings}
            write_started = time.perf_counter()
            if recorder:
                recorder.write(frames, views, depth)
            timings["record"] = (time.perf_counter() - write_started) * 1000
            quit_requested = False
            if not args.headless:
                for name, view in views.items():
                    cv2.imshow(name, view)
                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")
            timings["total"] = (time.perf_counter() - started) * 1000 + capture_ms
            stats.add(timings)
            if recorder:
                recorder.write_event(event)
            if quit_requested or (args.max_frames and frames >= args.max_frames):
                break
            capture_started = time.perf_counter()
            ret, frame = cap.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000
            if not ret or frame is None or not frame.size:
                if args.source is None:
                    print("[WARN] Потерян кадр с камеры; обработка остановлена")
                break
    finally:
        try:
            if recorder:
                recorder.close(stats.summary())
        finally:
            cap.release()
            if not args.headless:
                cv2.destroyAllWindows()
    summary = stats.summary()
    print(f"[INFO] Обработано кадров: {frames}")
    return summary
