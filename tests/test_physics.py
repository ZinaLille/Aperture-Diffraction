"""Physics, split and stage-one pipeline checks."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aperture.apertures import STRATEGIES, make_aperture, split_ids
from aperture.audit import (detector_indices, physical_metrics,
                            integrate_reference, fraunhofer_cell_raw)
from aperture.physics import (PhysicsConfig, check_nf, geometry, postprocess,
                              angular_spectrum_raw, rayleigh_sommerfeld_points)
from aperture.utils import new_output

class PipelineTests(unittest.TestCase):
    def test_nf_limits(self):
        for values in ([0], [5.01], [float("nan")], [1, 1]):
            with self.assertRaises(ValueError): check_nf(values)
        self.assertEqual(check_nf([0.01, 5]), (0.01, 5.0))

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            output = new_output(Path(folder)/"new")
            marker = output/"keep.txt"
            marker.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError): new_output(output)
            self.assertEqual(marker.read_text(), "untouched")

    def test_import_does_not_create_data(self):
        root = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(Path(__file__).resolve().parent))
            env = dict(os.environ, PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-c", "import aperture.apertures; import pathlib; assert not pathlib.Path('data').exists()"],
                                    cwd=folder, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_geometry_and_shared_splits(self):
        cfg = PhysicsConfig(grid=64)
        rng = np.random.RandomState(42)
        strategies = list(STRATEGIES)*4
        splits = split_ids(strategies)
        self.assertEqual(set.union(*(set(v) for v in splits.values())), set(range(32)))
        for k in splits:
            for j in splits:
                if j != k: self.assertFalse(set(splits[k]) & set(splits[j]))
        for name in STRATEGIES:
            a = make_aperture(name, rng, 64)
            self.assertEqual(set(np.unique(a)), {0, 1})
            geo = geometry(a, 5, cfg)
            self.assertAlmostEqual(geo["D_m"]**2/(cfg.wavelength_m*geo["z_m"]), 5)
            self.assertAlmostEqual(geo["detector_pitch_m"], cfg.wavelength_m*geo["z_m"]/(64*cfg.input_pitch_m))

    def test_reference_against_independent_integral(self):
        cfg = PhysicsConfig(grid=64)
        yy, xx = np.indices((64, 64))
        a = ((xx-32)**2+(yy-32)**2 <= 10**2).astype(float)
        geo = geometry(a, 5, cfg)
        raw = angular_spectrum_raw(a, geo["z_m"], cfg)
        ids = np.array([[32, 32], [32, 33], [32, 35], [35, 35]])
        direct = rayleigh_sommerfeld_points(a, geo["z_m"], (ids[:, ::-1]-32)*geo["detector_pitch_m"], cfg, 4)
        self.assertLess(np.max(np.abs(direct-raw[ids[:, 0], ids[:, 1]]))/direct.max(), 0.01)
        self.assertTrue(np.isfinite(raw).all())

    def test_zero_residual_and_physics_gradient(self):
        import torch
        from aperture.models import ForwardModel
        torch.manual_seed(42)
        model = ForwardModel("physics", base_ch=8)
        a = torch.rand(2, 1, 32, 32, requires_grad=True)
        pred, simulated, delta = model(a, torch.zeros(2, 2))
        self.assertEqual(torch.count_nonzero(delta).item(), 0)
        self.assertTrue(torch.equal(pred, simulated))
        pred.mean().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0)
        with torch.no_grad():
            expected = postprocess(abs(np.fft.fftshift(np.fft.fft2(a[0, 0].detach().numpy())))**2)
            self.assertLess(np.mean(abs(expected-simulated[0, 0].numpy())), 1e-6)

    def test_torch_asm_matches_numpy_and_has_gradients(self):
        import torch
        from aperture.physics import AngularSpectrumDetectorLayer
        torch.set_num_threads(4)
        cfg = PhysicsConfig(grid=32)
        yy, xx = np.indices((32, 32))
        a = ((xx-16)**2+(yy-16)**2 <= 5**2).astype(np.float32)
        z = geometry(a, 3, cfg)["z_m"]
        expected = postprocess(angular_spectrum_raw(a, z, cfg))
        for precision, tolerance in (("float32", 1e-6), ("float64", 1e-8)):
            layer = AngularSpectrumDetectorLayer(cfg, precision)
            value = torch.tensor(a)[None, None].requires_grad_()
            distance = torch.tensor([z], dtype=torch.float64, requires_grad=True)
            predicted = layer(value, distance)
            self.assertLess(np.mean(abs(predicted.detach().numpy()[0, 0]-expected)), tolerance)
            predicted.mean().backward()
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertTrue(torch.isfinite(distance.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
            self.assertEqual(sum(p.numel() for p in layer.parameters()), 0)
            self.assertEqual(len(layer.state_dict()), 0)
        with self.assertRaises(ValueError): layer(value, None)
        with self.assertRaises(ValueError): layer(value, [-1])

    def test_stage_timer_sync_boundaries(self):
        from aperture.utils import StageTimer
        with patch("aperture.utils.synchronize") as sync:
            with StageTimer("cuda") as timer:
                sum(range(1000))
            self.assertEqual(sync.call_count, 2)
            self.assertGreaterEqual(timer.seconds, 0)


class AuditTests(unittest.TestCase):
    def test_common_scale_preserves_amplitude_error(self):
        ref = np.array([0., 1., .5])
        self.assertEqual(physical_metrics(ref, ref)["linear_mae_ref_peak"], 0)
        self.assertAlmostEqual(physical_metrics(2*ref, ref)["linear_mae_ref_peak"], .5)
        self.assertGreater(physical_metrics(2*ref, ref)["common_log_mae"], 0)

    def test_point_coordinates_cover_peaks_and_bounds(self):
        value = np.zeros((32,32)); value[0,31] = 1
        points = detector_indices(32, (value,))
        self.assertEqual(len(points), len(set(map(tuple, points))))
        self.assertTrue(((points >= 0)&(points < 32)).all())
        self.assertIn((16,16), set(map(tuple, points)))
        self.assertIn((0,31), set(map(tuple, points)))

    def test_rs_single_pixel_far_field_matches_cell_sinc(self):
        cfg = PhysicsConfig(grid=32)
        a = np.zeros((32,32)); a[16,16] = 1
        z = geometry(a, .001, cfg)["z_m"]
        offsets = np.array([[0,0],[3,0],[0,7],[10,10]])
        xy = offsets*cfg.wavelength_m*z/(32*cfg.input_pitch_m)
        # A single pixel has strong high-frequency content: c32 is deliberately
        # insufficient at these outer points. Require the gate to reject it.
        _, _, coarse_passed = integrate_reference(a,z,xy,cfg,32,5e-4,2e-4)
        self.assertFalse(coarse_passed)
        ref, steps, passed = integrate_reference(a,z,xy,cfg,64,5e-4,2e-4)
        expected = fraunhofer_cell_raw(a,z,cfg)[16+offsets[:,1],16+offsets[:,0]]
        self.assertTrue(passed)
        self.assertGreaterEqual(len(steps), 3)
        self.assertLess(np.max(abs(ref[-1]-expected))/expected.max(), 5e-4)


if __name__ == "__main__":
    unittest.main()
