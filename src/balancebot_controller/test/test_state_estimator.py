#!/usr/bin/env python3
"""Unit tests for StateEstimator sensor fusion and TeleopNode."""

import math
import unittest
import numpy as np

from balancebot_controller.state_estimator import StateEstimator


class TestStateEstimator(unittest.TestCase):
    """Test suite for state estimation, IMU fusion, and encoder odometry."""

    def test_state_estimator_imu_pitch_extraction(self):
        """Verify IMU quaternion and complementary filter extract pitch angle correctly."""
        estimator = StateEstimator(wheel_radius=0.10)

        pitch_target = math.radians(10.0)
        qw = math.cos(pitch_target / 2.0)
        qy = math.sin(pitch_target / 2.0)
        quat = (0.0, qy, 0.0, qw)

        estimator.update_imu(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.05, 0.0), quat=quat)
        self.assertTrue(math.isclose(estimator.theta, pitch_target, abs_tol=1e-3))
        self.assertTrue(math.isclose(estimator.theta_dot, 0.05, abs_tol=1e-4))

    def test_state_estimator_wheel_encoder_fusion(self):
        """Verify forward linear velocity and position are calculated from wheel joint states."""
        estimator = StateEstimator(wheel_radius=0.10, nominal_length=0.28)

        names = ["left_wheel_joint", "right_wheel_joint"]
        positions = [10.0, 10.0]     # 10 rad each -> 10 * 0.10 = 1.0 m
        velocities = [5.0, 5.0]      # 5 rad/s each -> 5 * 0.10 = 0.5 m/s

        estimator.update_joints(names, positions, velocities)
        p, v, theta, theta_dot = estimator.get_state()

        self.assertTrue(math.isclose(estimator.wheel_p, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(estimator.wheel_v, 0.5, abs_tol=1e-5))
        self.assertTrue(math.isclose(p, 1.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(v, 0.5, abs_tol=1e-5))

    def test_state_estimator_leg_length_kinematics(self):
        """Verify effective leg length calculated from hip and knee joint angles."""
        estimator = StateEstimator(l1=0.20, l2=0.20, nominal_length=0.28)

        names = ["left_hip_joint", "left_knee_joint", "right_hip_joint", "right_knee_joint"]
        positions = [0.0, 0.0, 0.0, 0.0]
        velocities = [0.0, 0.0, 0.0, 0.0]

        estimator.update_joints(names, positions, velocities)
        self.assertTrue(math.isclose(estimator.effective_length, 0.38, abs_tol=1e-3))

        positions_crouch = [0.5, 1.0, 0.5, 1.0]
        estimator.update_joints(names, positions_crouch, velocities)
        self.assertTrue(0.18 <= estimator.effective_length <= 0.22)

    def test_state_estimator_complementary_filter_fallback(self):
        """Verify complementary filter converges to true pitch when quaternion is None."""
        estimator = StateEstimator(wheel_radius=0.10, alpha_complementary=0.98)
        true_pitch = math.radians(6.0)
        ax = 9.81 * math.sin(true_pitch)
        az = 9.81 * math.cos(true_pitch)
        for _ in range(150):
            estimator.update_imu(accel=(ax, 0.0, az), gyro=(0.0, 0.0, 0.0), quat=None, dt=0.01)
        self.assertTrue(math.isclose(estimator.theta, true_pitch, abs_tol=0.01))

    def test_state_estimator_noisy_imu_processing(self):
        """Verify StateEstimator filters out accelerometer Gaussian noise without diverging."""
        estimator = StateEstimator(wheel_radius=0.10, alpha_complementary=0.98)
        np.random.seed(42)
        true_pitch = 0.0
        filtered_pitches = []
        raw_pitches = []
        for _ in range(200):
            noise_ax = float(np.random.normal(0.0, 0.05))
            noise_az = float(np.random.normal(0.0, 0.05))
            noise_gy = float(np.random.normal(0.0, 0.01))
            ax = 9.81 * math.sin(true_pitch) + noise_ax
            az = 9.81 * math.cos(true_pitch) + noise_az
            raw_pitches.append(math.atan2(ax, az))
            estimator.update_imu(accel=(ax, 0.0, az), gyro=(0.0, noise_gy, 0.0), quat=None, dt=0.01)
            filtered_pitches.append(estimator.theta)

        self.assertTrue(math.isclose(estimator.theta, true_pitch, abs_tol=0.03))
        self.assertLessEqual(np.std(filtered_pitches[-100:]), np.std(raw_pitches[-100:]))

    def test_state_estimator_node_dynamic_dt_and_quat_validation(self):
        """Verify StateEstimatorNode calculates dynamic dt from stamp and handles quat validity."""
        import rclpy
        from sensor_msgs.msg import Imu
        from balancebot_controller.state_estimator import StateEstimatorNode

        if not rclpy.ok():
            rclpy.init()
        try:
            node = StateEstimatorNode()
            msg = Imu()
            msg.header.stamp.sec = 10
            msg.header.stamp.nanosec = 0
            msg.linear_acceleration.z = 9.81
            msg.orientation_covariance[0] = -1.0  # Invalid orientation covariance
            node._imu_callback(msg)
            self.assertEqual(node.last_imu_stamp, 10.0)

            # Second callback 5 ms later (200 Hz IMU rate)
            msg.header.stamp.sec = 10
            msg.header.stamp.nanosec = 5_000_000
            msg.angular_velocity.y = 0.2
            node._imu_callback(msg)
            self.assertAlmostEqual(node.last_imu_stamp, 10.005)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
