"""合成模型测试本地转换；真实许可数据不进入测试夹具。"""
import io
from pathlib import Path
import sys
import unittest
import zipfile

try:
    import numpy as np
except ImportError:
    np = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
if np is not None:
    import amass_local_skeleton as local


@unittest.skipIf(np is None, "AMASS 本地实验需要单独的 NumPy 环境")
class LocalSkeletonTests(unittest.TestCase):
    def synthetic(self):
        vertices = np.zeros((52, 3))
        vertices[:, 0] = np.arange(52)
        tree = np.array([[0] * 52, list(range(52))])
        model = {"v_template": vertices, "shapedirs": np.zeros((52, 3, 16)),
                 "J_regressor": np.eye(52), "kintree_table": tree}
        motion = {"poses": np.zeros((6, 156)), "trans": np.zeros((6, 3)), "betas": np.zeros(16)}
        return motion, model

    def test_identity_maps_x_metres_to_centimetres(self):
        motion, model = self.synthetic()
        joints, frames, selected = local.skeleton_frames(motion, model)
        self.assertEqual(selected, list(range(6)))
        self.assertEqual(len(joints), 22)
        self.assertEqual(frames[0][3:6], [100.0, 0.0, 0.0])

    def test_bad_shapes_and_nonfinite_values_are_rejected(self):
        motion, model = self.synthetic()
        motion["poses"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            local.skeleton_frames(motion, model)
        motion, model = self.synthetic()
        model["kintree_table"][0, 3] = 3
        with self.assertRaises(ValueError):
            local.skeleton_frames(motion, model)

    def test_npz_object_array_cannot_be_loaded(self):
        buf = io.BytesIO()
        np.savez(buf, poses=np.array([{"unsafe": True}], dtype=object))
        with self.assertRaises(ValueError):
            local.read_npz(buf.getvalue(), ("poses",))

    def test_npz_duplicate_member_is_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.npy", b"one")
            with self.assertWarns(UserWarning):
                z.writestr("a.npy", b"two")
        with self.assertRaises(ValueError):
            local.read_npz(buf.getvalue(), ("a",))


if __name__ == "__main__":
    unittest.main()
