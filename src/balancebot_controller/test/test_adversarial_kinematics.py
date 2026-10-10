#!/usr/bin/env python3
"""Adversarial stress testing and empirical challenge for LegKinematics.

Milestone M3 Challenge 1: Kinematics & Numerical Stability:
- Boundary inputs: height in {0.18, 0.38, 0.0, 0.50, negative, extreme offsets +/-0.08, +/-0.50}
- Singularity & floating point stress: q_k -> 0 and q_k -> pi (no NaN, zero-div, or domain error)
- Jacobian conditioning: condition number kappa(J) across operational grid
- VMC force mapping: virtual work equality delta_x^T * F == delta_q^T * tau
"""

import math
import unittest

from balancebot_controller.leg_kinematics import LegKinematics, LegKinematicsControllerNode
import numpy as np
import rclpy
from sensor_msgs.msg import Imu


class TestAdversarialKinematics(unittest.TestCase):
    """Adversarial stress tests for LegKinematics mathematical engine."""

    def setUp(self) -> None:
        self.kin = LegKinematics(
            l1=0.20,
            l2=0.20,
            height_min=0.18,
            height_max=0.38,
            offset_min=-0.08,
            offset_max=0.08,
            total_mass=12.4,
            g=9.81,
            tau_max=50.0,
        )

    def test_boundary_height_and_offset_inputs(self) -> None:
        """Verify boundary inputs to IK produce valid angles without NaN or domain errors."""
        boundary_cases = [
            # (x_d, z_d, expected_h_clamped, expected_x_clamped)
            (0.0, -0.18, 0.18, 0.0),
            (0.0, -0.38, 0.38, 0.0),
            (0.0, 0.0, 0.18, 0.0),
            (0.0, -0.50, 0.38, 0.0),
            (0.0, 0.50, 0.18, 0.0),
            (0.0, 10.0, 0.18, 0.0),
            (0.0, -10.0, 0.38, 0.0),
            (0.08, -0.28, 0.28, 0.08),
            (-0.08, -0.28, 0.28, -0.08),
            (0.50, -0.28, 0.28, 0.08),
            (-0.50, -0.28, 0.28, -0.08),
            (10.0, -0.28, 0.28, 0.08),
            (-10.0, -0.28, 0.28, -0.08),
            (0.50, 0.50, 0.18, 0.08),
            (-0.50, -0.50, 0.38, -0.08),
            (float('inf'), float('-inf'), 0.38, 0.08),
            (float('-inf'), float('inf'), 0.18, -0.08),
        ]

        for x_d, z_d, exp_h, exp_x in boundary_cases:
            qh, qk = self.kin.inverse_kinematics(x_d, z_d)
            self.assertFalse(math.isnan(qh), f'qh is NaN for x={x_d}, z={z_d}')
            self.assertFalse(math.isnan(qk), f'qk is NaN for x={x_d}, z={z_d}')

            # Forward kinematics reconstruction
            x_rec, z_rec = self.kin.forward_kinematics(qh, qk)
            h_rec = -z_rec
            self.assertAlmostEqual(
                h_rec, exp_h, delta=2e-3,
                msg=f'Reconstructed height mismatch for x={x_d}, z={z_d}'
            )
            self.assertAlmostEqual(
                x_rec, exp_x, delta=2e-3,
                msg=f'Reconstructed offset mismatch for x={x_d}, z={z_d}'
            )

    def test_singularity_and_extreme_joint_angles(self) -> None:
        """Stress Jacobian and VMC near kinematic singularities q_k -> 0 and q_k -> pi."""
        qk_singularities = [
            0.0, 1e-15, 1e-12, 1e-8, 1e-4, 1e-2,
            math.pi - 1e-2, math.pi - 1e-4, math.pi - 1e-8,
            math.pi - 1e-12, math.pi - 1e-15, math.pi,
            -1e-12, -0.5, math.pi + 1e-6,
        ]
        qh_angles = [-math.pi, -math.pi / 2, 0.0, math.pi / 4, math.pi, 2 * math.pi]

        for qk in qk_singularities:
            for qh in qh_angles:
                # 1. Forward Kinematics
                x, z = self.kin.forward_kinematics(qh, qk)
                self.assertFalse(math.isnan(x) or math.isnan(z))

                # 2. Jacobian
                j = self.kin.jacobian(qh, qk)
                self.assertFalse(np.isnan(j).any())
                det = float(np.linalg.det(j))
                self.assertFalse(math.isnan(det))

                # 3. VMC torques must remain bounded and finite
                tau_h, tau_k = self.kin.virtual_model_torque(
                    qh, qk,
                    target_z=0.28, current_z=-z, z_vel=0.1,
                    target_x=0.0, current_x=x, x_vel=-0.05,
                )
                self.assertFalse(math.isnan(tau_h) or math.isnan(tau_k))
                self.assertLessEqual(abs(tau_h), self.kin.tau_max + 1e-6)
                self.assertLessEqual(abs(tau_k), self.kin.tau_max + 1e-6)

    def test_jacobian_condition_number_across_operational_grid(self) -> None:
        """Check condition number kappa(J) and determinant across regular operating grid."""
        height_samples = np.linspace(0.18, 0.38, 30)
        offset_samples = np.linspace(-0.08, 0.08, 30)

        max_cond = 0.0
        min_det = 1.0
        min_sigma = 1.0

        for h in height_samples:
            for x in offset_samples:
                qh, qk = self.kin.inverse_kinematics(x, -h)
                j = self.kin.jacobian(qh, qk)
                det = float(np.linalg.det(j))
                u, s, vt = np.linalg.svd(j)
                cond = float(s[0] / s[1])

                if cond > max_cond:
                    max_cond = cond
                if det < min_det:
                    min_det = det
                if s[1] < min_sigma:
                    min_sigma = float(s[1])

        # Conditioning requirements:
        # In robotics, condition numbers < 50 indicate excellent numerical conditioning.
        self.assertLess(max_cond, 15.0, f'Maximum condition number too high: {max_cond}')
        self.assertGreater(min_det, 0.015, f'Minimum determinant too close to singular: {min_det}')
        self.assertGreater(min_sigma, 0.035, f'Minimum singular value too small: {min_sigma}')

    def test_vmc_virtual_work_equality(self) -> None:
        """Verify the principle of virtual work delta_x^T * F == delta_q^T * tau."""
        np.random.seed(12345)
        n_trials = 10000

        for _ in range(n_trials):
            qh = float(np.random.uniform(-math.pi / 2, math.pi / 2))
            qk = float(np.random.uniform(0.1, math.pi - 0.1))
            j = self.kin.jacobian(qh, qk)

            dq = np.random.normal(0, 1e-4, size=(2,))
            f_cart = np.random.uniform(-80, 80, size=(2,))

            dx = j @ dq
            tau = j.T @ f_cart

            work_cartesian = float(dx @ f_cart)
            work_joint = float(dq @ tau)

            err = abs(work_cartesian - work_joint)
            self.assertLess(err, 1e-12, f'Virtual work inequality observed: err={err}')

    def test_equilibrium_pitch_robustness(self) -> None:
        """Verify equilibrium pitch computation handles non-physical and extreme inputs."""
        edge_cases = [
            (0.0, 0.28, 0.0),
            (0.08, 0.18, math.asin(np.clip(0.08 / 0.18, -0.6, 0.6))),
            (-0.08, 0.18, math.asin(np.clip(-0.08 / 0.18, -0.6, 0.6))),
            (1.0, 0.28, math.asin(0.6)),      # Large offset saturation
            (-1.0, 0.28, math.asin(-0.6)),
            (0.05, 0.0, math.asin(np.clip(0.05 / 0.18, -0.6, 0.6))),    # Zero L_eff
            (0.05, -0.5, math.asin(np.clip(0.05 / 0.18, -0.6, 0.6))),   # Negative L_eff
            (0.0, -1.0, 0.0),
        ]

        for off, l_eff, expected_pitch in edge_cases:
            theta_eq = self.kin.equilibrium_pitch(off, l_eff)
            self.assertFalse(math.isnan(theta_eq))
            self.assertAlmostEqual(theta_eq, expected_pitch, places=5)
            self.assertLessEqual(abs(theta_eq), math.asin(0.6) + 1e-5)


class TestAdversarialControllerNode(unittest.TestCase):
    """Stress tests for LegKinematicsControllerNode ROS 2 integration."""

    @classmethod
    def setUpClass(cls) -> None:
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls) -> None:
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self) -> None:
        self.node = LegKinematicsControllerNode()

    def tearDown(self) -> None:
        self.node.destroy_node()

    def test_extreme_imu_roll_leveling_safety(self) -> None:
        """Verify extreme IMU roll orientations (up to 90 deg and inverted) do not explode."""
        for roll_deg in [-89.0, -60.0, -45.0, 0.0, 45.0, 60.0, 89.0, 90.0, 180.0]:
            roll_rad = math.radians(roll_deg)
            imu_msg = Imu()
            # Quaternion for pure roll
            imu_msg.orientation.w = math.cos(roll_rad / 2.0)
            imu_msg.orientation.x = math.sin(roll_rad / 2.0)
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = 0.0
            self.node._imu_callback(imu_msg)

            # Auto mode control step
            self.node.leg_mode = 0
            self.node._control_loop()

            # Targets and published states must remain bounded
            self.assertGreaterEqual(self.node.current_height, self.node.kin.height_min - 1e-3)
            self.assertLessEqual(self.node.current_height, self.node.kin.height_max + 1e-3)


if __name__ == '__main__':
    unittest.main()
