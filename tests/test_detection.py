import unittest
from types import SimpleNamespace

import numpy as np
import torch

from gpo.detection import object_distances


class DetectionTests(unittest.TestCase):
    def test_uses_inner_box_depth_and_excludes_invalid(self):
        box = SimpleNamespace(xyxy=torch.tensor([[0, 0, 20, 20]]),
                              cls=torch.tensor([0]), conf=torch.tensor([0.9]))
        result = SimpleNamespace(boxes=[box], names={0: "person"})
        depth = np.full((20, 20), 10.0)
        depth[4:16, 4:16] = 2.0
        self.assertEqual(object_distances(result, depth)[0]["depth_m"], 2.0)
        depth[:] = np.nan
        self.assertIsNone(object_distances(result, depth)[0]["depth_m"])
