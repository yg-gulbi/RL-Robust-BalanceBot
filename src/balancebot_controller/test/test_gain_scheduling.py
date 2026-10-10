#!/usr/bin/env python3
"""Unit tests for LQR Gain Scheduling and ARE solver."""

import math
import unittest
import numpy as np
import scipy.linalg

from balancebot_controller.balance_controller import GainScheduledLQRController


class TestGainScheduling(unittest.TestCase):
    """Test suite for continuous ARE solver and gain scheduling across heights."""

    def test_are_solution_existence_and_stability(self):
        """Verify Algebraic Riccati Equation yields strictly Hurwitz closed-loop poles."""
        ctrl = GainScheduledLQRController()

        lengths = [0.18, 0.23, 0.28, 0.33, 0.38]
        for length in lengths:
            m_total = ctrl.m_total_effective
            delta = m_total * (ctrl.I_b + ctrl.m_b * length**2) - (ctrl.m_b * length)**2

            a_mat = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -(ctrl.m_b * length)**2 * ctrl.g / delta, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, m_total * ctrl.m_b * ctrl.g * length / delta, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ])
            b_mat = np.array([
                [0.0],
                [(2.0 * (ctrl.I_b + ctrl.m_b * length**2) / ctrl.R + ctrl.m_b * length) / delta],
                [0.0],
                [-(2.0 * ctrl.m_b * length / ctrl.R + m_total) / delta],
                [0.0],
            ])

            p_sol = scipy.linalg.solve_continuous_are(a_mat, b_mat, ctrl.q_mat, ctrl.r_mat)
            # Solution P must be symmetric positive-semidefinite
            self.assertTrue(np.allclose(p_sol, p_sol.T, atol=1e-8))
            eigvals_p = np.linalg.eigvalsh(p_sol)
            self.assertTrue(np.all(eigvals_p >= -1e-8))

            # Closed-loop matrix A_cl = A - B * K
            k_gains = (np.linalg.inv(ctrl.r_mat) @ b_mat.T @ p_sol).reshape(1, -1)
            a_cl = a_mat - b_mat @ k_gains
            eigvals_cl = np.linalg.eigvals(a_cl)

            # Real parts of closed-loop poles:
            # Active controlled poles (velocity, pitch, pitch rate, integrator) are Hurwitz
            real_parts = np.sort(np.real(eigvals_cl))
            self.assertTrue(np.all(real_parts[:4] < -0.05), f"Active poles not Hurwitz: {real_parts}")
            self.assertLess(abs(real_parts[4]), 1e-10, f"Position pole not near zero: {real_parts[4]}")

    def test_gain_grid_structure_and_dimensions(self):
        """Verify gain grid contains required length range and 5 feedback gains."""
        ctrl = GainScheduledLQRController(height_min=0.18, height_max=0.38, num_grid_points=21)

        self.assertEqual(len(ctrl.gain_grid), 21)
        for length, gains in ctrl.gain_grid.items():
            self.assertTrue(0.179 <= length <= 0.381)
            self.assertEqual(len(gains), 5)
            self.assertTrue(np.all(np.isfinite(gains)))
            self.assertGreater(abs(gains[2]), 10.0)  # k_theta
            self.assertGreater(abs(gains[3]), 1.0)   # k_theta_dot

    def test_gain_interpolation_monotonicity_and_continuity(self):
        """Verify gain interpolation is continuous across heights [0.18, 0.38] m."""
        ctrl = GainScheduledLQRController()

        gains_low = ctrl.get_gains(0.18)
        gains_mid = ctrl.get_gains(0.28)
        gains_high = ctrl.get_gains(0.38)

        self.assertIsNotNone(gains_low)
        self.assertIsNotNone(gains_mid)
        self.assertIsNotNone(gains_high)

        # Continuity test
        test_heights = np.linspace(0.18, 0.38, 50)
        prev_gains = ctrl.get_gains(test_heights[0])
        for h in test_heights[1:]:
            curr_gains = ctrl.get_gains(h)
            diff = np.max(np.abs(curr_gains - prev_gains))
            self.assertLess(diff, 2.0, f"Discontinuous jump {diff} at height {h}")
            prev_gains = curr_gains

    def test_gain_clamping_outside_workspace(self):
        """Verify heights outside [0.18, 0.38] are clamped safely to boundary gains."""
        ctrl = GainScheduledLQRController()

        g_min = ctrl.get_gains(0.18)
        g_max = ctrl.get_gains(0.38)

        g_below = ctrl.get_gains(0.10)
        g_above = ctrl.get_gains(0.50)

        self.assertTrue(np.allclose(g_below, g_min, atol=1e-6))
        self.assertTrue(np.allclose(g_above, g_max, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
