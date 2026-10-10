#!/usr/bin/env python3
"""Unit tests for actuator torque saturation limits and integrator anti-windup."""

import math
import unittest

from balancebot_controller.balance_controller import GainScheduledLQRController


class TestAntiWindup(unittest.TestCase):
    """Test suite for actuator torque saturation limits and integrator anti-windup."""

    def test_actuator_torque_saturation_limits_25nm(self):
        """Verify wheel torque command clamps strictly at +/- 25.0 N*m limit."""
        ctrl = GainScheduledLQRController(tau_max=25.0)

        # Large positive pitch perturbation
        tau, sat = ctrl.compute_torque(
            v_meas=0.0,
            v_ref=0.0,
            theta_meas=0.9,
            theta_ref=0.0,
            theta_dot=0.0,
            e_integral=0.0,
            effective_length=0.28,
        )
        self.assertTrue(math.isclose(abs(tau), 25.0, abs_tol=1e-5))
        self.assertTrue(sat)

        # Large negative pitch perturbation
        tau_neg, sat_neg = ctrl.compute_torque(
            v_meas=0.0,
            v_ref=0.0,
            theta_meas=-0.9,
            theta_ref=0.0,
            theta_dot=0.0,
            e_integral=0.0,
            effective_length=0.28,
        )
        self.assertTrue(math.isclose(tau_neg, -25.0, abs_tol=1e-5))
        self.assertTrue(sat_neg)

    def test_torque_within_linear_bounds_not_saturated(self):
        """Verify small disturbance produces linear torque without saturation."""
        ctrl = GainScheduledLQRController(tau_max=25.0)

        tau, sat = ctrl.compute_torque(
            v_meas=0.0,
            v_ref=0.0,
            theta_meas=0.02,
            theta_ref=0.0,
            theta_dot=0.0,
            e_integral=0.0,
            effective_length=0.28,
        )
        self.assertLess(abs(tau), 25.0)
        self.assertFalse(sat)

    def test_anti_windup_freezes_integrator(self):
        """Verify anti-windup freezes integrator accumulation during torque saturation."""
        ctrl = GainScheduledLQRController(tau_max=25.0)

        e_integral = 1.0
        dt = 0.01

        # In saturation: integrator should freeze
        e_updated_sat = ctrl.update_integrator(
            e_integral=e_integral,
            v_meas=0.0,
            v_ref=5.0,
            dt=dt,
            saturated=True,
        )
        self.assertEqual(e_updated_sat, e_integral, "Integrator must freeze when saturated")

        # Out of saturation: integrator should accumulate
        e_updated_unsat = ctrl.update_integrator(
            e_integral=e_integral,
            v_meas=0.0,
            v_ref=1.0,
            dt=dt,
            saturated=False,
        )
        expected = e_integral + (0.0 - 1.0) * dt
        self.assertTrue(math.isclose(e_updated_unsat, expected, abs_tol=1e-6))

    def test_configurable_torque_limits(self):
        """Verify controller adapts to custom torque limit (e.g. 15.0 N*m)."""
        ctrl_15 = GainScheduledLQRController(tau_max=15.0)
        tau, sat = ctrl_15.compute_torque(
            v_meas=0.0,
            v_ref=0.0,
            theta_meas=0.8,
            theta_ref=0.0,
            theta_dot=0.0,
            e_integral=0.0,
            effective_length=0.28,
        )
        self.assertTrue(math.isclose(abs(tau), 15.0, abs_tol=1e-5))
        self.assertTrue(sat)


if __name__ == "__main__":
    unittest.main()
