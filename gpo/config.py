import argparse
import math
import json
from pathlib import Path
from camera import add_camera_args, validate_camera_args
from . import __version__


def parse_args(argv=None, default_mode="depth"):
    parser = argparse.ArgumentParser(
        description="RGB camera -> monocular depth -> pseudo-LiDAR / obstacle map"
    )

    add_camera_args(parser)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--mode", choices=["detect", "depth", "combined"], default=default_mode)
    parser.add_argument("--config", type=Path, help="JSON с настройками; CLI имеет приоритет")
    parser.add_argument("--weights", type=Path,
                        default=Path(__file__).resolve().parents[1] / "yolo11m.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--camera-height", type=float, help="Высота камеры над полом, метры")
    parser.add_argument("--pitch-deg", type=float, default=0, help="Наклон вниз, градусы")
    parser.add_argument("--min-height", type=float, default=0.05)
    parser.add_argument("--max-height", type=float, default=2.0)
    parser.add_argument("--roi", type=float, nargs=4, default=[0.35, 0.55, 0.65, 0.95],
                        metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--min-component-pixels", type=int, default=25)
    parser.add_argument("--release-frames", type=int, default=3)
    parser.add_argument("--source", type=Path, help="Видеофайл вместо камеры")
    parser.add_argument("--headless", action="store_true", help="Не открывать окна")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = до конца видео/выхода")
    parser.add_argument("--output-dir", type=Path, help="Каталог для записей отдельных запусков")
    parser.add_argument("--output-fps", type=float, default=30, help="Частота воспроизведения сохранённого видео")
    parser.add_argument("--save-depth", action="store_true", help="Сохранять метрическую глубину в NPY")

    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        help="Название модели глубины HuggingFace"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство для инференса"
    )

    parser.add_argument(
        "--infer-width",
        type=int,
        default=518,
        help="Ширина изображения для нейросети. Меньше = быстрее"
    )

    parser.add_argument(
        "--max-depth",
        type=float,
        default=20.0,
        help="Максимальная глубина в метрах для визуализации"
    )

    parser.add_argument(
        "--stop-distance",
        type=float,
        default=1.2,
        help="Дистанция стоп-зоны в метрах"
    )

    parser.add_argument(
        "--slow-distance",
        type=float,
        default=2.5,
        help="Дистанция зоны замедления в метрах"
    )

    parser.add_argument(
        "--danger-ratio",
        type=float,
        default=0.08,
        help="Порог доли близких пикселей в ROI, после которого путь считается опасным"
    )

    parser.add_argument(
        "--fov-deg",
        type=float,
        default=70.0,
        help="Примерный горизонтальный FOV камеры в градусах, если нет калибровки"
    )

    parser.add_argument(
        "--fx",
        type=float,
        default=None,
        help="Фокусное расстояние камеры fx в пикселях"
    )

    parser.add_argument(
        "--fy",
        type=float,
        default=None,
        help="Фокусное расстояние камеры fy в пикселях"
    )

    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Главная точка камеры cx"
    )

    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Главная точка камеры cy"
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Шаг выборки точек для pseudo-LiDAR"
    )

    parser.add_argument(
        "--side-range",
        type=float,
        default=5.0,
        help="Ширина pseudo-LiDAR карты влево/вправо от камеры, метры"
    )

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    config_path = pre_parser.parse_known_args(argv)[0].config
    if config_path:
        try:
            defaults = read_config(config_path, parser)
            parser.set_defaults(**defaults)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    args = parser.parse_args(argv)
    try:
        validate_camera_args(args)
        validate_depth_args(args)
        if not 0 < args.confidence < 1 or args.imgsz <= 0:
            raise ValueError("Нужны 0 < confidence < 1 и imgsz > 0")
        if args.max_frames < 0 or not math.isfinite(args.output_fps) or args.output_fps <= 0:
            raise ValueError("Нужны max-frames >= 0 и конечный output-fps > 0")
        if args.save_depth and not args.output_dir:
            raise ValueError("save-depth требует output-dir")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def read_config(path, parser):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Корень конфигурации должен быть JSON-объектом")
    actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "config", "version")}
    for key, value in data.items():
        if key not in actions:
            raise ValueError(f"Неизвестная настройка: {key}")
        action = actions[key]
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError(f"{key} должен быть true или false")
            continue
        if value is None:
            if action.default is not None:
                raise ValueError(f"{key} не допускает null")
            continue
        if action.nargs:
            if not isinstance(value, list) or len(value) != action.nargs:
                raise ValueError(f"{key}: нужно {action.nargs} значений")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                raise ValueError(f"{key}: нужны числа")
            data[key] = [action.type(v) for v in value]
            continue
        if action.type is int and type(value) is not int:
            raise ValueError(f"{key} должен быть целым числом")
        if action.type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"{key} должен быть числом")
        if action.type in (str, Path) and value is not None and not isinstance(value, str):
            raise ValueError(f"{key} должен быть строкой")
        if action.choices and value not in action.choices:
            raise ValueError(f"Недопустимое значение {key}: {value}")
        elif action.type:
            data[key] = action.type(value)
            if action.type is Path and not data[key].is_absolute():
                data[key] = path.resolve().parent / data[key]
    return data


def validate_depth_args(args):
    x1, y1, x2, y2 = args.roi
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("ROI: нужны 0 <= x1 < x2 <= 1 и 0 <= y1 < y2 <= 1")
    if not 0 < args.min_valid_ratio <= 1:
        raise ValueError("min-valid-ratio должен быть в (0, 1]")
    if args.min_component_pixels <= 0 or args.release_frames <= 0:
        raise ValueError("min-component-pixels и release-frames должны быть > 0")
    if args.camera_height is not None and (not math.isfinite(args.camera_height) or args.camera_height <= 0):
        raise ValueError("camera-height должна быть конечным числом > 0")
    if not -90 < args.pitch_deg < 90:
        raise ValueError("pitch-deg должен быть между -90 и 90")
    if not math.isfinite(args.min_height) or not math.isfinite(args.max_height) or args.min_height >= args.max_height:
        raise ValueError("Нужно min-height < max-height, оба значения конечны")
    for name in ("infer_width", "stride"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} должен быть > 0")
    for name in ("max_depth", "stop_distance", "slow_distance", "side_range"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} должен быть конечным числом > 0")
    if not 0.1 < args.stop_distance < args.slow_distance < args.max_depth:
        raise ValueError("Нужно 0.1 < stop-distance < slow-distance < max-depth")
    if not 0 < args.danger_ratio < 1:
        raise ValueError("danger-ratio должен быть между 0 и 1")
    if not 0 < args.fov_deg < 180:
        raise ValueError("fov-deg должен быть между 0 и 180")
    for name in ("fx", "fy", "cx", "cy"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or (name in ("fx", "fy") and value <= 0)):
            raise ValueError(f"Некорректный параметр камеры {name}")
    if args.side_range * 70 < 1:
        raise ValueError("side-range слишком мал для карты")
