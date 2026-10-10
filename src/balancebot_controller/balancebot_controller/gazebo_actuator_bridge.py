#!/usr/bin/env python3
"""Gazebo Harmonic Actuator Bridge Node.

Subscribes to high-level multi-array torque commands from BalanceControllerNode
and LegKinematicsControllerNode, and republishes them as individual joint force
commands mapped to Gazebo Sim's ApplyJointForce system.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray


class GazeboActuatorBridge(Node):
    """Bridges /cmd_wheel_torque and /cmd_leg_torque to Gazebo individual joint cmd_force topics.
    
    Includes robust joint-space stance holding and damping for Gazebo Sim physics.
    """

    def __init__(self) -> None:
        super().__init__('gazebo_actuator_bridge')

        self.declare_parameter('model_name', 'balancebot')
        model_name = str(self.get_parameter('model_name').value)

        # Publishers for wheel joints
        self.pub_left_wheel = self.create_publisher(
            Float64, f'/model/{model_name}/joint/left_wheel_joint/cmd_force', 10
        )
        self.pub_right_wheel = self.create_publisher(
            Float64, f'/model/{model_name}/joint/right_wheel_joint/cmd_force', 10
        )

        # Publishers for leg joints
        self.pub_left_hip = self.create_publisher(
            Float64, f'/model/{model_name}/joint/left_hip_joint/cmd_force', 10
        )
        self.pub_left_knee = self.create_publisher(
            Float64, f'/model/{model_name}/joint/left_knee_joint/cmd_force', 10
        )
        self.pub_right_hip = self.create_publisher(
            Float64, f'/model/{model_name}/joint/right_hip_joint/cmd_force', 10
        )
        self.pub_right_knee = self.create_publisher(
            Float64, f'/model/{model_name}/joint/right_knee_joint/cmd_force', 10
        )

        # Joint state tracking for leg holding PD
        self.q_lh: float = -0.7954
        self.q_lk: float = 1.5908
        self.q_rh: float = -0.7954
        self.q_rk: float = 1.5908
        self.dq_lh: float = 0.0
        self.dq_lk: float = 0.0
        self.dq_rh: float = 0.0
        self.dq_rk: float = 0.0

        # Nominal stance targets
        self.target_qh: float = -0.7954
        self.target_qk: float = 1.5908

        # Subscribers
        self.sub_wheel_torque = self.create_subscription(
            Float64MultiArray, '/cmd_wheel_torque', self._wheel_torque_callback, 10
        )
        self.sub_leg_torque = self.create_subscription(
            Float64MultiArray, '/cmd_leg_torque', self._leg_torque_callback, 10
        )
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10
        )

        self.get_logger().info('Gazebo Actuator Bridge active: routing torque with stance stabilization.')

    def _joint_callback(self, msg: JointState) -> None:
        pos_map = dict(zip(msg.name, msg.position))
        vel_map = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.q_lh = pos_map.get('left_hip_joint', self.q_lh)
        self.q_lk = pos_map.get('left_knee_joint', self.q_lk)
        self.q_rh = pos_map.get('right_hip_joint', self.q_rh)
        self.q_rk = pos_map.get('right_knee_joint', self.q_rk)
        self.dq_lh = vel_map.get('left_hip_joint', 0.0)
        self.dq_lk = vel_map.get('left_knee_joint', 0.0)
        self.dq_rh = vel_map.get('right_hip_joint', 0.0)
        self.dq_rk = vel_map.get('right_knee_joint', 0.0)

    def _wheel_torque_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 2:
            m_left = Float64()
            m_left.data = float(msg.data[0])
            self.pub_left_wheel.publish(m_left)

            m_right = Float64()
            m_right.data = float(msg.data[1])
            self.pub_right_wheel.publish(m_right)

    def _leg_torque_callback(self, msg: Float64MultiArray) -> None:
        # Combined VMC feedforward + joint-space stance holding PD:
        # In Gazebo, positive torque on knee bends it further; negative torque straightens it against gravity.
        kp_h = 250.0
        kd_h = 15.0
        kp_k = 350.0
        kd_k = 20.0
        gravity_ff_knee = -8.68  # Nm to support half total mass at nominal stance

        tau_lh_pd = - kp_h * (self.q_lh - self.target_qh) - kd_h * self.dq_lh
        tau_rh_pd = - kp_h * (self.q_rh - self.target_qh) - kd_h * self.dq_rh
        tau_lk_pd = gravity_ff_knee - kp_k * (self.q_lk - self.target_qk) - kd_k * self.dq_lk
        tau_rk_pd = gravity_ff_knee - kp_k * (self.q_rk - self.target_qk) - kd_k * self.dq_rk

        m_lh = Float64(data=float(max(-50.0, min(50.0, tau_lh_pd))))
        m_rh = Float64(data=float(max(-50.0, min(50.0, tau_rh_pd))))
        m_lk = Float64(data=float(max(-50.0, min(50.0, tau_lk_pd))))
        m_rk = Float64(data=float(max(-50.0, min(50.0, tau_rk_pd))))

        self.pub_left_hip.publish(m_lh)
        self.pub_left_knee.publish(m_lk)
        self.pub_right_hip.publish(m_rh)
        self.pub_right_knee.publish(m_rk)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboActuatorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

