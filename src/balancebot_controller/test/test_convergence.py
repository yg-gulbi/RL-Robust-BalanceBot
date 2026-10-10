#!/usr/bin/env python3
"""Unit tests verifying closed-loop balance stability, convergence, and settling time."""

import math
import unittest
import numpy as np

from balancebot_controller.balance_controller import GainScheduledLQRController


class WIPMSimulator:
    """Numerical simulation of Wheeled Inverted Pendulum Model dynamics."""

    def __init__(self, l: float = 0.28, tau_max: float = 25.0) -> None:
        self.l = l
        self.m_b = 10.0
        self.M_w = 1.2
        self.R = 0.10
        self.I_b = 0.135
        self.I_w = 0.006
        self.g = 9.81
        self.tau_max = tau_max

        self.m_total = self.m_b + 2.0 * self.M_w + (2.0 * self.I_w) / (self.R**2)

        # States
        self.x = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.e_integral = 0.0
        self.t = 0.0

    def step(self, tau_w: float, dt: float = 0.001, ext_force: float = 0.0) -> None:
        """Advance WIPM dynamic equations by one time step."""
        det_m = self.m_total * (self.I_b + self.m_b * self.l**2) - (self.m_b * self.l * math.cos(self.theta))**2
        det_m = max(det_m, 1e-4)

        c_p = (
            self.m_b * self.l * (self.theta_dot**2) * math.sin(self.theta)
            + (2.0 / self.R) * tau_w
            + ext_force
            - 0.5 * self.v
        )
        c_theta = (
            self.m_b * self.g * self.l * math.sin(self.theta)
            - tau_w
            - 0.05 * self.theta_dot
        )

        inv11 = (self.I_b + self.m_b * self.l**2) / det_m
        inv12 = -(self.m_b * self.l * math.cos(self.theta)) / det_m
        inv21 = inv12
        inv22 = self.m_total / det_m

        accel_p = inv11 * c_p + inv12 * c_theta
        accel_theta = inv21 * c_p + inv22 * c_theta

        self.v += accel_p * dt
        self.x += self.v * dt
        self.theta_dot += accel_theta * dt
        self.theta += self.theta_dot * dt
        self.t += dt


class TestConvergence(unittest.TestCase):
    """Test suite for closed-loop pitch balance, velocity convergence, and settling time."""

    def test_flat_ground_steady_state_pitch_balance(self):
        """Verify robot maintains pitch balance within |theta| < 5.0 deg (0.087 rad)."""
        sim = WIPMSimulator(l=0.28)
        sim.theta = math.radians(3.0)  # Initial 3 deg tilt

        ctrl = GainScheduledLQRController(tau_max=25.0)

        pitches = []
        dt = 0.001
        for step in range(3000):  # 3.0 seconds
            tau, sat = ctrl.compute_torque(
                v_meas=sim.v,
                v_ref=0.0,
                theta_meas=sim.theta,
                theta_ref=0.0,
                theta_dot=sim.theta_dot,
                e_integral=sim.e_integral,
                effective_length=sim.l,
            )
            sim.e_integral = ctrl.update_integrator(sim.e_integral, sim.v, 0.0, dt, sat)
            sim.step(tau, dt=dt)
            if step >= 2000:  # steady state regime (last 1.0s)
                pitches.append(abs(math.degrees(sim.theta)))

        self.assertGreater(len(pitches), 0)
        max_steady_pitch = max(pitches)
        self.assertLess(max_steady_pitch, 5.0, f"Steady pitch {max_steady_pitch:.2f} deg exceeded 5.0 deg limit")

    def test_velocity_command_tracking_and_convergence(self):
        """Verify velocity converges to target with integral action and pitch remains stable."""
        sim = WIPMSimulator(l=0.28)
        ctrl = GainScheduledLQRController(tau_max=25.0)

        v_target = 0.8
        dt = 0.001
        vel_history = []
        pitch_history = []

        for step in range(5000):  # 5.0 seconds
            tau, sat = ctrl.compute_torque(
                v_meas=sim.v,
                v_ref=v_target,
                theta_meas=sim.theta,
                theta_ref=0.0,
                theta_dot=sim.theta_dot,
                e_integral=sim.e_integral,
                effective_length=sim.l,
            )
            sim.e_integral = ctrl.update_integrator(sim.e_integral, sim.v, v_target, dt, sat)
            sim.step(tau, dt=dt)
            if step >= 3500:  # Steady state window
                vel_history.append(sim.v)
                pitch_history.append(abs(math.degrees(sim.theta)))

        mean_vel = float(np.mean(vel_history))
        self.assertTrue(math.isclose(mean_vel, v_target, abs_tol=0.10))
        self.assertLess(max(pitch_history), 5.0)

    def test_settling_time_less_than_two_seconds(self):
        """Verify settling time T_s < 2.0 s from initial disturbance tilt."""
        sim = WIPMSimulator(l=0.28)
        sim.theta = math.radians(4.0)  # 4 degree initial tilt
        ctrl = GainScheduledLQRController(tau_max=25.0)

        dt = 0.001
        history = []

        for _ in range(4000):  # 4.0 seconds
            tau, sat = ctrl.compute_torque(
                v_meas=sim.v,
                v_ref=0.0,
                theta_meas=sim.theta,
                theta_ref=0.0,
                theta_dot=sim.theta_dot,
                e_integral=sim.e_integral,
                effective_length=sim.l,
            )
            sim.e_integral = ctrl.update_integrator(sim.e_integral, sim.v, 0.0, dt, sat)
            sim.step(tau, dt=dt)
            history.append((sim.t, abs(sim.v), abs(sim.theta)))

        # Tolerance band: v < 0.10 m/s and |theta| < 2.0 deg
        v_band = 0.10
        theta_band = math.radians(2.0)

        t_settled = history[-1][0]
        for i in range(len(history) - 1, -1, -1):
            t, v_err, th_err = history[i]
            if v_err > v_band or th_err > theta_band:
                t_settled = t
                break

        self.assertLess(t_settled, 2.0, f"Settling time {t_settled:.2f}s exceeded 2.0s limit")

    def test_gain_scheduled_balance_across_height_range(self):
        """Verify stable balance at low (0.18m), nominal (0.28m), and high (0.38m) heights."""
        ctrl = GainScheduledLQRController(tau_max=25.0)

        for height in [0.18, 0.28, 0.38]:
            sim = WIPMSimulator(l=height)
            sim.theta = math.radians(2.5)
            dt = 0.001

            for _ in range(2500):
                tau, sat = ctrl.compute_torque(
                    v_meas=sim.v,
                    v_ref=0.0,
                    theta_meas=sim.theta,
                    theta_ref=0.0,
                    theta_dot=sim.theta_dot,
                    e_integral=sim.e_integral,
                    effective_length=height,
                )
                sim.e_integral = ctrl.update_integrator(sim.e_integral, sim.v, 0.0, dt, sat)
                sim.step(tau, dt=dt)

            self.assertLess(abs(math.degrees(sim.theta)), 5.0, f"Height {height}m failed balance")
            self.assertLess(abs(sim.v), 0.15)


if __name__ == "__main__":
    unittest.main()
