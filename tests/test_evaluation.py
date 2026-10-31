import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_known_errors(self):
        with tempfile.TemporaryDirectory(dir=".venv") as folder:
            path = Path(folder) / "measurements.csv"
            path.write_text("true_m,estimated_m\n1,1.2\n2,1.8\n", encoding="utf-8")
            result = evaluate(path)
        self.assertAlmostEqual(result["mae_m"], 0.2)
        self.assertAlmostEqual(result["bias_m"], 0)
        self.assertEqual(result["samples"], 2)
