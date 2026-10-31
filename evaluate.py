"""Evaluate manually measured distances against depth estimates."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def evaluate(path):
    truth, estimated = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"true_m", "estimated_m"} <= set(reader.fieldnames):
            raise ValueError("CSV должен содержать true_m и estimated_m")
        for index, row in enumerate(reader, 2):
            try:
                actual, prediction = float(row["true_m"]), float(row["estimated_m"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Строка {index}: нужны численные расстояния") from exc
            if not all(math.isfinite(v) and v > 0 for v in (actual, prediction)):
                raise ValueError(f"Строка {index}: расстояния должны быть конечными и > 0")
            truth.append(actual)
            estimated.append(prediction)
    if not truth:
        raise ValueError("Нет измерений")
    errors = np.asarray(estimated) - truth
    return {"samples": len(truth), "mae_m": float(np.abs(errors).mean()),
            "bias_m": float(errors.mean()), "rmse_m": float(np.sqrt((errors ** 2).mean())),
            "abs_rel": float((np.abs(errors) / truth).mean()),
            "absolute_error_p95_m": float(np.percentile(np.abs(errors), 95))}


def main():
    parser = argparse.ArgumentParser(description="Оценка ошибок глубины по контрольным измерениям CSV")
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.measurements)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
