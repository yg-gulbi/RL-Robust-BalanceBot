#!/usr/bin/env python3
"""Gain-Scheduled LQR-I Balance Controller for BalanceBot.

Maintains longitudinal self-balance (|pitch| < 5.0 deg) and tracks velocity
commands at 100 Hz. Solves the continuous Algebraic Riccati Equation (ARE)
parameterized by effective leg length L in [0.18, 0.38] m.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.node import Node
import scipy.linalg
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64, Float64MultiArray, Int32

from .state_estimator import StateEstimator
from .terrain_observer import FlightState


class GainScheduledLQRController:
    """Gain-Scheduled LQR controller with Integral Action and Anti-Windup."""

    def __init__(
        self,
        m_b: float = 10.0,
        M_w: float = 1.2,
        R: float = 0.10,
        I_b: float = 0.135,
        I_w: float = 0.006,
        tau_max: float = 25.0,
        g: float = 9.81,
        q_diag: Tuple[float, float, float, float, float] = (0.0, 15.0, 80.0, 8.0, 5.0),
        r_cost: float = 1.0,
        height_min: float = 0.18,
        height_max: float = 0.38,
        num_grid_points: int = 21,
    ) -> None:
        self.m_b = m_b
        self.M_w = M_w
        self.R = R
        self.I_b = I_b
        self.I_w = I_w
        self.tau_max = tau_max
        self.g = g
        self.q_mat = np.diag(q_diag)
        self.r_mat = np.array([[r_cost]])
        self.height_min = height_min
        self.height_max = height_max
        self.num_grid_points = num_grid_points

        self.gain_grid: Dict[float, np.ndarray] = {}
        self._build_gain_grid()

    @property
    def m_total_effective(self) -> float:
        """Total effective mass including wheel inertia reflected to ground contact."""
        return self.m_b + 2.0 * self.M_w + (2.0 * self.I_w) / (self.R**2)

    def _build_gain_grid(self) -> None:
        """Precompute optimal LQR gains by solving continuous ARE across heights."""
        lengths = np.linspace(self.height_min, self.height_max, self.num_grid_points)
        m_total = self.m_total_effective

        for length in lengths:
            length_key = round(float(length), 3)
            delta = m_total * (self.I_b + self.m_b * length**2) - (self.m_b * length)**2
            delta = max(delta, 1e-5)

            # State-space linearized around theta=0:
            # x = [p, v, theta, theta_dot, e_I]^T
            a_mat = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -(self.m_b * length)**2 * self.g / delta, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, m_total * self.m_b * self.g * length / delta, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ], dtype=float)

            b_mat = np.array([
                [0.0],
                [(2.0 * (self.I_b + self.m_b * length**2) / self.R + self.m_b * length) / delta],
                [0.0],
                [-(2.0 * self.m_b * length / self.R + m_total) / delta],
                [0.0],
            ], dtype=float)

            # Continuous-time Algebraic Riccati Equation:
            # A^T P + P A - P B R^-1 B^T P + Q = 0
            p_sol = scipy.linalg.solve_continuous_are(a_mat, b_mat, self.q_mat, self.r_mat)
            k_gains = (np.linalg.inv(self.r_mat) @ b_mat.T @ p_sol).flatten()
            self.gain_grid[length_key] = k_gains

    def get_gains(self, effective_length: float) -> np.ndarray:
        """Linearly interpolate precomputed gains for effective leg length."""
        l_clamped = float(np.clip(effective_length, self.height_min, self.height_max))
        keys = sorted(self.gain_grid.keys())
        idx = int(np.searchsorted(keys, l_clamped))
        if idx == 0:
            return self.gain_grid[keys[0]]
        if idx >= len(keys):
            return self.gain_grid[keys[-1]]

        k_low, k_high = keys[idx - 1], keys[idx]
        alpha = (l_clamped - k_low) / (k_high - k_low)
        return (1.0 - alpha) * self.gain_grid[k_low] + alpha * self.gain_grid[k_high]

    def compute_torque(
        self,
        v_meas: float,
        v_ref: float,
        theta_meas: float,
        theta_ref: float,
        theta_dot: float,
        e_integral: float,
        effective_length: float,
        pos_meas: float = 0.0,
        pos_ref: float = 0.0,
    ) -> Tuple[float, bool]:
        """Compute drive wheel torque u and anti-windup saturation status."""
        gains = self.get_gains(effective_length)
        e_pos = pos_meas - pos_ref if pos_ref != 0.0 else 0.0
        e_vel = v_meas - v_ref
        e_theta = theta_meas - theta_ref

        state = np.array([e_pos, e_vel, e_theta, theta_dot, e_integral], dtype=float)
        tau_raw = -float(np.dot(gains, state))

        tau_clamped = float(np.clip(tau_raw, -self.tau_max, self.tau_max))
        saturated = bool(abs(tau_raw) >= self.tau_max)
        return tau_clamped, saturated

    def update_integrator(
        self,
        e_integral: float,
        v_meas: float,
        v_ref: float,
        dt: float,
        saturated: bool,
    ) -> float:
        """Integrate velocity error with anti-windup clamping."""
        if not saturated:
            return e_integral + (v_meas - v_ref) * dt
        # When saturated, freeze integrator accumulation
        return e_integral


class BalanceControllerNode(Node):
    """ROS 2 Node executing 100 Hz Gain-Scheduled LQR-I balance loop."""

    def __init__(self) -> None:
        super().__init__('balance_controller_node')

        # Declare parameters
        self.declare_parameter('control_rate', 100.0)
        self.declare_parameter('tau_max', 25.0)
        self.declare_parameter('torso_mass', 10.0)
        self.declare_parameter('wheel_mass', 1.2)
        self.declare_parameter('wheel_radius', 0.10)
        self.declare_parameter('k_omega', 8.0)
        self.declare_parameter('torque_topic', '/cmd_wheel_torque')

        rate = self.get_parameter('control_rate').value
        self.dt = 1.0 / rate
        tau_max = self.get_parameter('tau_max').value
        m_b = self.get_parameter('torso_mass').value
        m_w = self.get_parameter('wheel_mass').value
        wheel_r = self.get_parameter('wheel_radius').value
        self.k_omega = self.get_parameter('k_omega').value
        torque_topic = self.get_parameter('torque_topic').value

        # Initialize internal controller and estimator
        self.controller = GainScheduledLQRController(
            m_b=m_b,
            M_w=m_w,
            R=wheel_r,
            tau_max=tau_max,
        )
        self.estimator = StateEstimator(wheel_radius=wheel_r)

        # Reference commands
        self.v_ref: float = 0.0
        self.omega_ref: float = 0.0
        self.theta_cmd: float = 0.0
        self.theta_eq: float = 0.0
        self.theta_ref: float = 0.0
        self.e_integral: float = 0.0

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
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            10,
        )
        self.sub_robot_state = self.create_subscription(
            Float64MultiArray,
            '/balancebot/robot_state',
            self._robot_state_callback,
            10,
        )
        self.sub_leg_state = self.create_subscription(
            Float64MultiArray,
            '/balancebot/leg_state',
            self._leg_state_callback,
            10,
        )
        self.flight_state: FlightState = FlightState.GROUND_BALANCE
        self.sub_flight_state = self.create_subscription(
            Int32,
            '/balancebot/flight_state',
            self._flight_state_callback,
            10,
        )
        self.terrain_slope: float = 0.0
        self.sub_slope = self.create_subscription(
            Float64,
            '/balancebot/terrain_slope',
            self._slope_callback,
            10,
        )

        # Publishers
        self.pub_torque = self.create_publisher(Float64MultiArray, torque_topic, 10)
        self.pub_telemetry = self.create_publisher(Float64MultiArray, '/balancebot/telemetry', 10)

        # Control loop timer (100 Hz)
        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f'Balance Controller Node active (100 Hz, tau_max={tau_max} Nm, topic={torque_topic})'
        )

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.v_ref = float(msg.linear.x)
        self.omega_ref = float(msg.angular.z)

    def _imu_callback(self, msg: Imu) -> None:
        accel = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        gyro = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        quat = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        self.estimator.update_imu(accel, gyro, quat, dt=self.dt)

    def _joint_callback(self, msg: JointState) -> None:
        self.estimator.update_joints(list(msg.name), list(msg.position), list(msg.velocity))

    def _robot_state_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 4:
            self.estimator.p = float(msg.data[0])
            self.estimator.v = float(msg.data[1])
            self.estimator.theta = float(msg.data[2])
            self.estimator.theta_dot = float(msg.data[3])
            if len(msg.data) >= 5:
                self.estimator.effective_length = float(msg.data[4])
            if len(msg.data) >= 6:
                self.theta_eq = float(msg.data[5])
                self.theta_ref = self.theta_cmd + self.theta_eq

    def _leg_state_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 3:
            # msg.data = [L_eff, delta_x_offset, theta_eq, mode, roll]
            self.estimator.effective_length = float(msg.data[0])
            self.theta_eq = float(msg.data[2])
            self.theta_ref = self.theta_cmd + self.theta_eq

    def _flight_state_callback(self, msg: Int32) -> None:
        try:
            self.flight_state = FlightState(msg.data)
        except ValueError:
            self.flight_state = FlightState.GROUND_BALANCE

    def _slope_callback(self, msg: Float64) -> None:
        self.terrain_slope = float(msg.data)

    def _control_loop(self) -> None:
        # Retrieve current states
        p, v, theta, theta_dot = self.estimator.get_state()
        eff_l = self.estimator.effective_length
        yaw_rate = self.estimator.yaw_rate

        # Dynamically adjust reference pitch around equilibrium point with slope feedforward tilt
        self.theta_ref = self.theta_cmd + self.theta_eq + self.terrain_slope

        # Compute balance torque and anti-windup (suppressed in air & touchdown absorption)
        if self.flight_state in (FlightState.AIRBORNE, FlightState.TOUCHDOWN_ABSORPTION):
            tau_w = 0.0
            saturated = False
        else:
            tau_w, saturated = self.controller.compute_torque(
                v_meas=v,
                v_ref=self.v_ref,
                theta_meas=theta,
                theta_ref=self.theta_ref,
                theta_dot=theta_dot,
                e_integral=self.e_integral,
                effective_length=eff_l,
            )

            # Update integrator with anti-windup
            self.e_integral = self.controller.update_integrator(
                self.e_integral, v, self.v_ref, self.dt, saturated
            )

        # Differential steering torque for yaw
        delta_tau = self.k_omega * (self.omega_ref - yaw_rate)
        tau_l = float(np.clip(
            tau_w + 0.5 * delta_tau, -self.controller.tau_max, self.controller.tau_max
        ))
        tau_r = float(np.clip(
            tau_w - 0.5 * delta_tau, -self.controller.tau_max, self.controller.tau_max
        ))

        # Publish torque command: [tau_left, tau_right]
        msg = Float64MultiArray()
        msg.data = [tau_l, tau_r]
        self.pub_torque.publish(msg)

        # Publish telemetry: [v_meas, v_ref, theta_meas, theta_ref, tau_l, tau_r, e_int, eff_l]
        telem = Float64MultiArray()
        telem.data = [
            v, self.v_ref, theta, self.theta_ref,
            tau_l, tau_r, self.e_integral, eff_l, float(saturated),
        ]
        self.pub_telemetry.publish(telem)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = BalanceControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
