#!/usr/bin/env python3
"""State Estimator for BalanceBot.

Combines IMU (/imu/data) and wheel/leg encoders (/joint_states) into
the state vector x = [p, v, theta, theta_dot]^T along with effective
pendulum length L_eff.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray


class StateEstimator:
    """Core sensor fusion logic combining IMU and wheel encoders."""

    def __init__(
        self,
        wheel_radius: float = 0.10,
        l1: float = 0.20,
        l2: float = 0.20,
        nominal_length: float = 0.28,
        min_length: float = 0.18,
        max_length: float = 0.38,
        alpha_complementary: float = 0.98,
    ) -> None:
        self.r = wheel_radius
        self.l1 = l1
        self.l2 = l2
        self.nominal_length = nominal_length
        self.min_length = min_length
        self.max_length = max_length
        self.alpha = alpha_complementary

        # States
        self.p: float = 0.0          # longitudinal position (m)
        self.v: float = 0.0          # forward linear velocity (m/s)
        self.theta: float = 0.0      # torso pitch angle (rad)
        self.theta_dot: float = 0.0  # torso pitch angular velocity (rad/s)
        self.yaw: float = 0.0        # yaw angle (rad)
        self.yaw_rate: float = 0.0   # yaw rate (rad/s)
        self.effective_length: float = nominal_length

        # Internal tracking
        self.wheel_p: float = 0.0
        self.wheel_v: float = 0.0
        self.last_imu_time: Optional[float] = None
        self.initialized: bool = False

    def update_imu(
        self,
        accel: Tuple[float, float, float],
        gyro: Tuple[float, float, float],
        quat: Optional[Tuple[float, float, float, float]] = None,
        dt: float = 0.01,
    ) -> None:
        """Update orientation and rates from IMU accelerometer and gyroscope."""
        ax, ay, az = accel
        gx, gy, gz = gyro

        self.theta_dot = gy
        self.yaw_rate = gz

        if quat is not None and any(abs(q) > 1e-4 for q in quat):
            qx, qy, qz, qw = quat
            # Pitch from quaternion: rotation about Y
            sinp = 2.0 * (qw * qy - qz * qx)
            if abs(sinp) >= 1.0:
                pitch_quat = math.copysign(math.pi / 2.0, sinp)
            else:
                pitch_quat = math.asin(sinp)
            self.theta = pitch_quat
            # Yaw from quaternion
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
        else:
            # Complementary filter using accel + gyro
            norm_yz = math.sqrt(ay**2 + az**2)
            pitch_acc = math.atan2(ax, max(norm_yz, 1e-6))
            if not self.initialized:
                self.theta = pitch_acc
                self.initialized = True
            else:
                self.theta = self.alpha * (self.theta + gy * dt) + (1.0 - self.alpha) * pitch_acc
            self.yaw += gz * dt

    def update_joints(
        self,
        names: List[str],
        positions: List[float],
        velocities: List[float],
    ) -> None:
        """Update wheel odometry and leg kinematics from joint states."""
        name_pos: Dict[str, float] = dict(zip(names, positions))
        name_vel: Dict[str, float] = dict(zip(names, velocities))

        # 1. Wheels
        phi_l = name_pos.get('left_wheel_joint', 0.0)
        phi_r = name_pos.get('right_wheel_joint', 0.0)
        dphi_l = name_vel.get('left_wheel_joint', 0.0)
        dphi_r = name_vel.get('right_wheel_joint', 0.0)

        self.wheel_p = 0.5 * (phi_l + phi_r) * self.r
        self.wheel_v = 0.5 * (dphi_l + dphi_r) * self.r

        # 2. Leg length calculation from hip & knee
        lengths: List[float] = []
        for prefix in ('left', 'right'):
            qh_name = f'{prefix}_hip_joint'
            qk_name = f'{prefix}_knee_joint'
            if qh_name in name_pos and qk_name in name_pos:
                qh = name_pos[qh_name]
                qk = name_pos[qk_name]
                leg_len = self.l1 * math.cos(qh) + self.l2 * math.cos(qh + qk)
                lengths.append(leg_len)

        if lengths:
            avg_len = float(np.mean(lengths))
            self.effective_length = float(np.clip(avg_len, self.min_length, self.max_length))
        else:
            self.effective_length = self.nominal_length

        # 3. Robot CoM state vector computation
        self.p = self.wheel_p + self.effective_length * math.sin(self.theta)
        self.v = self.wheel_v + self.effective_length * self.theta_dot * math.cos(self.theta)

    def get_state(self) -> Tuple[float, float, float, float]:
        """Return state vector [p, v, theta, theta_dot]."""
        return self.p, self.v, self.theta, self.theta_dot

    def get_state_array(self) -> np.ndarray:
        """Return state vector as numpy array [p, v, theta, theta_dot]."""
        return np.array([self.p, self.v, self.theta, self.theta_dot], dtype=float)


class StateEstimatorNode(Node):
    """ROS 2 Node publishing fused robot state at 100 Hz."""

    def __init__(self) -> None:
        super().__init__('state_estimator_node')

        # Declare parameters
        self.declare_parameter('wheel_radius', 0.10)
        self.declare_parameter('nominal_length', 0.28)
        self.declare_parameter('publish_rate', 100.0)

        wheel_r = self.get_parameter('wheel_radius').value
        nom_l = self.get_parameter('nominal_length').value
        rate = self.get_parameter('publish_rate').value

        self.estimator = StateEstimator(wheel_radius=wheel_r, nominal_length=nom_l)
        self.last_imu_stamp: Optional[float] = None

        # Subscriptions
        self.sub_imu = self.create_subscription(
            Imu,
            '/imu/data',
            self._imu_callback,
            10,
        )
        self.sub_joints = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10,
        )

        # Publishers
        self.pub_state = self.create_publisher(Float64MultiArray, '/balancebot/robot_state', 10)
        self.pub_odom = self.create_publisher(Odometry, '/robot/state_estimate', 10)

        # Timer
        self.timer = self.create_timer(1.0 / rate, self._publish_callback)
        self.get_logger().info('State Estimator Node initialized (100 Hz).')

    def _imu_callback(self, msg: Imu) -> None:
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self.last_imu_stamp is not None and stamp_sec > self.last_imu_stamp:
            dt = float(np.clip(stamp_sec - self.last_imu_stamp, 0.001, 0.05))
        else:
            dt = 0.01
        self.last_imu_stamp = stamp_sec

        accel = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        gyro = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)

        # Validate quaternion presence: verify covariance flag and non-zero components
        quat_valid = (
            (len(msg.orientation_covariance) == 0 or msg.orientation_covariance[0] >= 0.0)
            and any(abs(q) > 1e-4 for q in (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))
        )
        quat = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w) if quat_valid else None
        self.estimator.update_imu(accel, gyro, quat, dt=dt)

    def _joint_callback(self, msg: JointState) -> None:
        self.estimator.update_joints(list(msg.name), list(msg.position), list(msg.velocity))

    def _publish_callback(self) -> None:
        p, v, theta, theta_dot = self.estimator.get_state()
        eff_l = self.estimator.effective_length

        # 1. Float64MultiArray: [p, v, theta, theta_dot, L_eff]
        state_msg = Float64MultiArray()
        state_msg.data = [p, v, theta, theta_dot, eff_l]
        self.pub_state.publish(state_msg)

        # 2. Odometry message
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = p
        odom.pose.pose.position.z = eff_l

        # Orientation quaternion from pitch & yaw
        cy = math.cos(self.estimator.yaw * 0.5)
        sy = math.sin(self.estimator.yaw * 0.5)
        cp = math.cos(theta * 0.5)
        sp = math.sin(theta * 0.5)
        odom.pose.pose.orientation.w = cy * cp
        odom.pose.pose.orientation.x = -sy * sp
        odom.pose.pose.orientation.y = cy * sp
        odom.pose.pose.orientation.z = sy * cp

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.y = theta_dot
        odom.twist.twist.angular.z = self.estimator.yaw_rate
        self.pub_odom.publish(odom)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
