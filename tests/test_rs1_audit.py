"""Checks for independent-reference sampling, scaling and convergence."""
import unittest
import numpy as np
from src.forward.rs1_audit import detector_indices, physical_metrics, integrate_reference, fraunhofer_cell_raw
from src.common.physics import PhysicsConfig, geometry


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
