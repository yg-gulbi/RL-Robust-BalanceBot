#!/usr/bin/env python3
"""Active 2-DOF Sagittal Leg Kinematics & Virtual Model Control for BalanceBot.

Implements Milestone M3 (Features F07-F12):
- Forward Kinematics (FK) and Inverse Kinematics (IK)
- Analytical Leg Jacobian and singularity-bounded mapping
- Virtual Model Impedance Control (VMC: tau = J^T * F) with gravity feedforward
- Dynamic Equilibrium Pitch Trim (theta_eq = arcsin(delta_x_offset / L_eff))
- Latched Manual Mode Tracking vs Auto Terrain Adaptation (Roll Leveling)
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray, Int32

from .terrain_observer import FlightState


class LegKinematics:
    """Pure mathematical formulation of 2-DOF sagittal leg kinematics and VMC."""

    def __init__(
        self,
        params: Optional[Any] = None,
        l1: float = 0.20,
        l2: float = 0.20,
        height_min: float = 0.18,
        height_max: float = 0.38,
        offset_min: float = -0.08,
        offset_max: float = 0.08,
        total_mass: float = 12.4,
        g: float = 9.81,
        tau_max: float = 50.0,
    ) -> None:
        if params is not None and hasattr(params, 'L1'):
            self.l1 = float(getattr(params, 'L1', l1))
            self.l2 = float(getattr(params, 'L2', l2))
            self.height_min = float(getattr(params, 'height_min', height_min))
            self.height_max = float(getattr(params, 'height_max', height_max))
            self.offset_min = float(getattr(params, 'offset_min', offset_min))
            self.offset_max = float(getattr(params, 'offset_max', offset_max))
            self.total_mass = float(getattr(params, 'total_mass', total_mass))
            self.g = float(getattr(params, 'g', g))
            self.tau_max = float(getattr(params, 'tau_max', tau_max))
        else:
            self.l1 = l1
            self.l2 = l2
            self.height_min = height_min
            self.height_max = height_max
            self.offset_min = offset_min
            self.offset_max = offset_max
            self.total_mass = total_mass
            self.g = g
            self.tau_max = tau_max

    @property
    def L1(self) -> float:
        return self.l1

    @property
    def L2(self) -> float:
        return self.l2

    def forward_kinematics(self, q_h: float, q_k: float) -> Tuple[float, float]:
        """Compute relative wheel center position (x_rel, z_rel) given hip and knee angles.

        Coordinates are relative to hip joint in sagittal plane:
        +X is forward, +Z is up (z_rel < 0 for leg pointing downward).
        """
        x_rel = self.l1 * math.sin(q_h) + self.l2 * math.sin(q_h + q_k)
        z_rel = -(self.l1 * math.cos(q_h) + self.l2 * math.cos(q_h + q_k))
        return float(x_rel), float(z_rel)

    def inverse_kinematics(self, x_d: float, z_d: float) -> Tuple[float, float]:
        """Compute joint angles (q_h, q_k) for desired foot position (x_d, z_d) with z_d < 0."""
        # 1. Clamp target within safe operational boundaries
        height_clamped = float(np.clip(-z_d, self.height_min, self.height_max))
        z_clamped = -height_clamped
        x_clamped = float(np.clip(x_d, self.offset_min, self.offset_max))

        # 2. Law of Cosines for knee angle q_k
        dist_sq = x_clamped**2 + z_clamped**2
        d_val = (dist_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        d_val = float(np.clip(d_val, -0.98, 0.95))
        q_k = math.acos(d_val)

        # 3. Hip angle q_h
        alpha = math.atan2(x_clamped, -z_clamped)
        beta = math.atan2(self.l2 * math.sin(q_k), self.l1 + self.l2 * math.cos(q_k))
        q_h = alpha - beta

        return float(q_h), float(q_k)

    def jacobian(self, q_h: float, q_k: float) -> np.ndarray:
        """Compute 2x2 sagittal leg Jacobian matrix J(q) = d(x, z) / d(q_h, q_k)."""
        j11 = self.l1 * math.cos(q_h) + self.l2 * math.cos(q_h + q_k)
        j12 = self.l2 * math.cos(q_h + q_k)
        j21 = self.l1 * math.sin(q_h) + self.l2 * math.sin(q_h + q_k)
        j22 = self.l2 * math.sin(q_h + q_k)
        return np.array([[j11, j12], [j21, j22]], dtype=float)

    def virtual_model_torque(
        self,
        q_h: float,
        q_k: float,
        target_z: float,
        current_z: float,
        z_vel: float = 0.0,
        k_z: float = 1200.0,
        d_z: float = 80.0,
        f_gravity_ff: Optional[float] = None,
        target_x: float = 0.0,
        current_x: float = 0.0,
        x_vel: float = 0.0,
        k_x: float = 800.0,
        d_x: float = 40.0,
    ) -> Tuple[float, float]:
        """Map Cartesian virtual spring-damper wrench into joint torques tau = J^T * F."""
        if f_gravity_ff is None:
            f_gravity_ff = (self.total_mass * self.g) / 2.0

        if target_z > 0 and current_z > 0:
            f_z = k_z * (target_z - current_z) - d_z * z_vel + f_gravity_ff
        else:
            f_z = k_z * (target_z - current_z) + d_z * (-z_vel) + f_gravity_ff

        f_x = k_x * (target_x - current_x) - d_x * x_vel

        f_cartesian = np.array([f_x, f_z], dtype=float)
        j = self.jacobian(q_h, q_k)
        tau = j.T @ f_cartesian
        tau_h = float(np.clip(tau[0], -self.tau_max, self.tau_max))
        tau_k = float(np.clip(tau[1], -self.tau_max, self.tau_max))
        return tau_h, tau_k

    def equilibrium_pitch(self, offset_x: float, effective_length: float) -> float:
        """Compute dynamic equilibrium pitch angle theta_eq = arcsin(delta_x_offset / L_eff)."""
        eff_l = max(effective_length, self.height_min)
        ratio = float(np.clip(offset_x / eff_l, -0.6, 0.6))
        return float(math.asin(ratio))


class LegKinematicsControllerNode(Node):
    """ROS 2 Node executing 50 Hz active leg regulation loop."""

    def __init__(self) -> None:
        super().__init__('leg_kinematics_controller_node')

        # Parameters
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('total_mass', 12.4)
        self.declare_parameter('track_width', 0.40)
        self.declare_parameter('k_z', 1200.0)
        self.declare_parameter('d_z', 80.0)
        self.declare_parameter('k_x', 800.0)
        self.declare_parameter('d_x', 40.0)
        self.declare_parameter('nominal_height', 0.28)
        self.declare_parameter('tau_max', 50.0)

        rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / rate
        total_m = float(self.get_parameter('total_mass').value)
        self.track_w = float(self.get_parameter('track_width').value)
        self.k_z = float(self.get_parameter('k_z').value)
        self.d_z = float(self.get_parameter('d_z').value)
        self.k_x = float(self.get_parameter('k_x').value)
        self.d_x = float(self.get_parameter('d_x').value)
        self.nom_h = float(self.get_parameter('nominal_height').value)
        self.tau_max = float(self.get_parameter('tau_max').value)

        self.kin = LegKinematics(total_mass=total_m, tau_max=self.tau_max)

        # State tracking (Mode 0: AUTO_ADAPTATION, Mode 1: MANUAL_TARGET)
        self.leg_mode: int = 0
        self.current_height: float = self.nom_h
        self.current_offset: float = 0.0
        self.target_height: float = self.nom_h
        self.target_offset: float = 0.0
        self.roll_angle: float = 0.0

        # Joint measurements: [left_hip, left_knee, right_hip, right_knee]
        self.q_lh: float = 0.0
        self.q_lk: float = 0.0
        self.q_rh: float = 0.0
        self.q_rk: float = 0.0
        self.dq_lh: float = 0.0
        self.dq_lk: float = 0.0
        self.dq_rh: float = 0.0
        self.dq_rk: float = 0.0

        # Subscriptions
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10
        )
        self.sub_imu = self.create_subscription(
            Imu, '/imu/data', self._imu_callback, 10
        )
        self.sub_cmd_mode = self.create_subscription(
            Float64MultiArray, '/set_leg_mode', self._cmd_mode_callback, 10
        )
        self.sub_cmd_mode_alt = self.create_subscription(
            Float64MultiArray, '/balancebot/cmd_leg_mode', self._cmd_mode_callback, 10
        )
        self.flight_state: FlightState = FlightState.GROUND_BALANCE
        self.sub_flight_state = self.create_subscription(
            Int32, '/balancebot/flight_state', self._flight_state_callback, 10
        )

        # Publishers
        self.pub_leg_joint_torque = self.create_publisher(
            JointState, '/cmd_leg_joint_torque', 10
        )
        self.pub_leg_torque = self.create_publisher(
            Float64MultiArray, '/cmd_leg_torque', 10
        )
        self.pub_leg_state = self.create_publisher(
            Float64MultiArray, '/balancebot/leg_state', 10
        )

        # 50 Hz control loop timer
        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info('Leg Kinematics Controller Node active (50 Hz).')

    def _joint_callback(self, msg: JointState) -> None:
        name_pos = dict(zip(msg.name, msg.position))
        name_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.q_lh = name_pos.get('left_hip_joint', self.q_lh)
        self.q_lk = name_pos.get('left_knee_joint', self.q_lk)
        self.q_rh = name_pos.get('right_hip_joint', self.q_rh)
        self.q_rk = name_pos.get('right_knee_joint', self.q_rk)
        self.dq_lh = name_vel.get('left_hip_joint', 0.0)
        self.dq_lk = name_vel.get('left_knee_joint', 0.0)
        self.dq_rh = name_vel.get('right_hip_joint', 0.0)
        self.dq_rk = name_vel.get('right_knee_joint', 0.0)

    def _imu_callback(self, msg: Imu) -> None:
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self.roll_angle = math.atan2(sinr_cosp, cosr_cosp)

    def _cmd_mode_callback(self, msg: Float64MultiArray) -> None:
        """Command format: [mode (0: auto, 1: manual), target_height, target_offset]."""
        if len(msg.data) >= 1:
            mode_val = int(msg.data[0])
            self.leg_mode = 1 if mode_val == 1 else 0
            if len(msg.data) >= 2:
                self.target_height = float(np.clip(
                    msg.data[1], self.kin.height_min, self.kin.height_max
                ))
            if len(msg.data) >= 3:
                self.target_offset = float(np.clip(
                    msg.data[2], self.kin.offset_min, self.kin.offset_max
                ))
            mode_str = 'MANUAL_TARGET' if self.leg_mode == 1 else 'AUTO_ADAPTATION'
            self.get_logger().info(
                f'Leg mode updated: {mode_str}, h_target={self.target_height:.3f}, '
                f'x_target={self.target_offset:.3f}'
            )

    def _flight_state_callback(self, msg: Int32) -> None:
        try:
            self.flight_state = FlightState(msg.data)
        except ValueError:
            self.flight_state = FlightState.GROUND_BALANCE

    def _control_loop(self) -> None:
        # 1. Forward kinematics of current joint measurements
        xl, zl = self.kin.forward_kinematics(self.q_lh, self.q_lk)
        xr, zr = self.kin.forward_kinematics(self.q_rh, self.q_rk)
        meas_hl = -zl
        meas_hr = -zr
        eff_l = 0.5 * (meas_hl + meas_hr)

        # 2. Mode-dependent target generation
        if self.leg_mode == 1:
            # Mode 1: MANUAL_TARGET with slew-rate limiting
            # (|dh| <= 0.20 m/s, |dx| <= 0.15 m/s)
            dh = float(np.clip(
                self.target_height - self.current_height, -0.20 * self.dt, 0.20 * self.dt
            ))
            dx = float(np.clip(
                self.target_offset - self.current_offset, -0.15 * self.dt, 0.15 * self.dt
            ))
            self.current_height += dh
            self.current_offset += dx
            h_left = self.current_height
            h_right = self.current_height
            offset_left = self.current_offset
            offset_right = self.current_offset
            k_z_active = self.k_z
            d_z_active = self.d_z
        else:
            # Mode 0: AUTO_ADAPTATION modulated by FlightState
            if self.flight_state == FlightState.RAMP_ASCENT:
                h_base = 0.22
                max_rate = 0.50
                k_z_active = 1200.0
                d_z_active = 80.0
            elif self.flight_state == FlightState.AIRBORNE:
                h_base = 0.36
                max_rate = 0.50
                k_z_active = 600.0
                d_z_active = 40.0
            elif self.flight_state == FlightState.TOUCHDOWN_ABSORPTION:
                h_base = 0.20
                max_rate = 0.50
                k_z_active = 400.0
                d_z_active = 250.0
            elif self.flight_state == FlightState.BALANCE_RECOVERY:
                h_base = self.nom_h
                max_rate = 0.20
                k_z_active = 1200.0
                d_z_active = 80.0
            else:  # GROUND_BALANCE
                h_base = self.nom_h
                max_rate = 0.20
                k_z_active = self.k_z
                d_z_active = self.d_z

            dh = float(np.clip(
                h_base - self.current_height, -max_rate * self.dt, max_rate * self.dt
            ))
            self.current_height += dh
            self.current_offset = 0.0

            delta_h = (self.track_w / 2.0) * math.tan(self.roll_angle)
            h_left = float(np.clip(
                self.current_height + delta_h, self.kin.height_min, self.kin.height_max
            ))
            h_right = float(np.clip(
                self.current_height - delta_h, self.kin.height_min, self.kin.height_max
            ))
            offset_left = 0.0
            offset_right = 0.0

        # 3. Cartesian velocities via Jacobian
        jl = self.kin.jacobian(self.q_lh, self.q_lk)
        jr = self.kin.jacobian(self.q_rh, self.q_rk)
        vl = jl @ np.array([self.dq_lh, self.dq_lk], dtype=float)
        vr = jr @ np.array([self.dq_rh, self.dq_rk], dtype=float)
        # zl_dot = vl[1], so height_dot = -zl_dot
        z_vel_l = -vl[1]
        z_vel_r = -vr[1]

        # 4. Virtual Model Control joint torques
        tau_lh, tau_lk = self.kin.virtual_model_torque(
            self.q_lh, self.q_lk,
            target_z=h_left, current_z=meas_hl, z_vel=z_vel_l,
            target_x=offset_left, current_x=xl, x_vel=vl[0],
            k_z=k_z_active, d_z=d_z_active, k_x=self.k_x, d_x=self.d_x,
        )
        tau_rh, tau_rk = self.kin.virtual_model_torque(
            self.q_rh, self.q_rk,
            target_z=h_right, current_z=meas_hr, z_vel=z_vel_r,
            target_x=offset_right, current_x=xr, x_vel=vr[0],
            k_z=k_z_active, d_z=d_z_active, k_x=self.k_x, d_x=self.d_x,
        )

        # 5. Dynamic pitch equilibrium shift
        theta_eq = self.kin.equilibrium_pitch(self.current_offset, eff_l)

        # 6. Publish JointState effort commands: /cmd_leg_joint_torque
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = [
            'left_hip_joint', 'left_knee_joint',
            'right_hip_joint', 'right_knee_joint'
        ]
        js_msg.effort = [tau_lh, tau_lk, tau_rh, tau_rk]
        self.pub_leg_joint_torque.publish(js_msg)

        # Also publish Float64MultiArray for direct array subscribers: /cmd_leg_torque
        arr_msg = Float64MultiArray()
        arr_msg.data = [tau_lh, tau_lk, tau_rh, tau_rk]
        self.pub_leg_torque.publish(arr_msg)

        # 7. Publish leg telemetry: /balancebot/leg_state [L_eff, offset_x, theta_eq, mode, roll]
        state_msg = Float64MultiArray()
        state_msg.data = [
            float(eff_l),
            float(self.current_offset),
            float(theta_eq),
            float(self.leg_mode),
            float(self.roll_angle),
        ]
        self.pub_leg_state.publish(state_msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = LegKinematicsControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
