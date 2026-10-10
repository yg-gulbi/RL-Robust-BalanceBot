#!/usr/bin/env python3
"""Terrain Observer and 5-State Flight/Landing FSM for BalanceBot.

Implements Milestone M4 (Features F13-F17):
- FlightState IntEnum (GROUND_BALANCE, RAMP_ASCENT, AIRBORNE, TOUCHDOWN_ABSORPTION, BALANCE_RECOVERY)
- Free-fall specific force thresholding (||a|| < 2.5 m/s^2 for >= 30 ms)
- Touchdown impact spike detection (||a|| > 15.0 m/s^2)
- Touchdown absorption dissipation (|z_dot| < 0.05 m/s or timeout > 0.25 s)
- Balance recovery stabilization (|theta| < 5 deg and |z_dot| < 0.03 m/s)
- Ground slope estimation from pitch and accelerometer
"""

from __future__ import annotations

import enum
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64, Int32


class FlightState(enum.IntEnum):
    """5-State discrete flight and landing finite state machine."""

    GROUND_BALANCE = 0
    RAMP_ASCENT = 1
    AIRBORNE = 2
    TOUCHDOWN_ABSORPTION = 3
    BALANCE_RECOVERY = 4


class TerrainObserver:
    """5-State flight/landing FSM and ground slope estimator."""

    def __init__(
        self,
        freefall_accel_thresh: float = 2.5,
        impact_accel_thresh: float = 15.0,
        debounce_freefall_s: float = 0.030,
        absorption_settle_vel: float = 0.05,
        absorption_timeout_s: float = 0.25,
        recovery_pitch_thresh: float = 0.08726,  # 5.0 deg (math.radians(5.0))
        recovery_settle_vel: float = 0.03,
        ramp_pitch_thresh: float = 0.15,
        nominal_height: float = 0.28,
        airborne_height: float = 0.36,
        absorption_height: float = 0.20,
        ramp_height: float = 0.22,
    ) -> None:
        self.state: FlightState = FlightState.GROUND_BALANCE
        self.freefall_accel_thresh = freefall_accel_thresh
        self.impact_accel_thresh = impact_accel_thresh
        self.debounce_freefall_s = debounce_freefall_s
        self.absorption_settle_vel = absorption_settle_vel
        self.absorption_timeout_s = absorption_timeout_s
        self.recovery_pitch_thresh = recovery_pitch_thresh
        self.recovery_settle_vel = recovery_settle_vel
        self.ramp_pitch_thresh = ramp_pitch_thresh

        self.nom_h = nominal_height
        self.air_h = airborne_height
        self.abs_h = absorption_height
        self.ramp_h = ramp_height

        self.freefall_timer: float = 0.0
        self.recovery_timer: float = 0.0
        self.touchdown_time: Optional[float] = None
        self.airborne_time: Optional[float] = None

        self.terrain_slope: float = 0.0
        self.slope_estimate: float = 0.0

    def estimate_slope(self, pitch: float, ax: float = 0.0, az: float = 9.81) -> float:
        """Estimate terrain slope from IMU pitch and accelerometer components."""
        if abs(az) > 1e-3:
            slope_raw = math.atan2(ax, az)
        else:
            slope_raw = pitch
        slope = 0.5 * slope_raw + 0.5 * pitch
        return float(slope)

    def update(
        self,
        t: float,
        accel_mag: float,
        wheel_contact: Union[bool, float] = True,
        pitch: float = 0.0,
        z_vel: float = 0.0,
        dt: float = 0.02,
        pitch_rate: float = 0.0,
        v_meas: float = 0.0,
        ax: float = 0.0,
        az: float = 9.81,
    ) -> FlightState:
        """Update FSM state based on IMU specific force, kinematics, and contact."""
        # Handle calling convention where pitch is passed as 3rd positional argument
        if isinstance(wheel_contact, (float, int)) and not isinstance(wheel_contact, bool):
            actual_pitch = float(wheel_contact)
            actual_contact = True
            actual_z_vel = float(pitch)
            actual_dt = float(z_vel) if z_vel != 0.0 else dt
        else:
            actual_contact = bool(wheel_contact)
            actual_pitch = float(pitch)
            actual_z_vel = float(z_vel)
            actual_dt = float(dt)

        # Update slope estimate when grounded
        if self.state in (FlightState.GROUND_BALANCE, FlightState.RAMP_ASCENT) and actual_contact:
            slope_raw = self.estimate_slope(actual_pitch, ax, az)
            alpha = 0.95
            self.terrain_slope = alpha * self.terrain_slope + (1.0 - alpha) * slope_raw
            self.slope_estimate = self.terrain_slope

        # State transitions
        if self.state == FlightState.GROUND_BALANCE:
            # Check ramp ascent condition (pitch tilt on ramp with forward motion)
            if v_meas > 0.15 and (actual_pitch > self.ramp_pitch_thresh or self.terrain_slope > self.ramp_pitch_thresh):
                self.state = FlightState.RAMP_ASCENT
            # Check free-fall
            elif not actual_contact or accel_mag < self.freefall_accel_thresh:
                self.freefall_timer += actual_dt
                if self.freefall_timer >= self.debounce_freefall_s:
                    self.state = FlightState.AIRBORNE
                    self.airborne_time = t
                    self.freefall_timer = 0.0
            else:
                self.freefall_timer = 0.0

        elif self.state == FlightState.RAMP_ASCENT:
            if not actual_contact or accel_mag < self.freefall_accel_thresh:
                self.state = FlightState.AIRBORNE
                self.airborne_time = t
            elif actual_pitch < 0.05 and v_meas > 0.1:
                # Level ground reached without jumping
                self.state = FlightState.GROUND_BALANCE

        elif self.state == FlightState.AIRBORNE:
            # Detect touchdown impact: require impact spike or contact restored with non-freefall specific force
            is_impact_spike = accel_mag > self.impact_accel_thresh
            is_contact_restored = actual_contact and accel_mag >= self.freefall_accel_thresh
            if is_impact_spike or is_contact_restored:
                self.state = FlightState.TOUCHDOWN_ABSORPTION
                self.touchdown_time = t
                self.recovery_timer = 0.0

        elif self.state == FlightState.TOUCHDOWN_ABSORPTION:
            # Impact absorbed when vertical velocity settles or timer expires
            time_in_absorption = (t - self.touchdown_time) if self.touchdown_time is not None else 0.0
            if abs(actual_z_vel) < self.absorption_settle_vel or time_in_absorption > self.absorption_timeout_s:
                self.state = FlightState.BALANCE_RECOVERY
                self.recovery_timer = 0.0

        elif self.state == FlightState.BALANCE_RECOVERY:
            self.recovery_timer += actual_dt
            if abs(actual_pitch) < self.recovery_pitch_thresh and abs(actual_z_vel) < self.recovery_settle_vel:
                self.state = FlightState.GROUND_BALANCE

        return self.state

    def get_control_targets(self) -> Tuple[float, float, float, bool]:
        """Return (target_height, k_z, d_z, suppress_torque)."""
        if self.state == FlightState.RAMP_ASCENT:
            return self.ramp_h, 1200.0, 80.0, False
        elif self.state == FlightState.AIRBORNE:
            return self.air_h, 600.0, 40.0, True
        elif self.state == FlightState.TOUCHDOWN_ABSORPTION:
            return self.abs_h, 400.0, 250.0, True
        elif self.state == FlightState.BALANCE_RECOVERY:
            return self.nom_h, 1200.0, 80.0, False
        return self.nom_h, 1200.0, 80.0, False


class TerrainObserverNode(Node):
    """ROS 2 Node executing 50 Hz terrain slope estimation and flight FSM."""

    def __init__(self) -> None:
        super().__init__('terrain_observer_node')

        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('freefall_accel_thresh', 2.5)
        self.declare_parameter('impact_accel_thresh', 15.0)
        self.declare_parameter('ramp_pitch_thresh', 0.15)

        rate = float(self.get_parameter('control_rate').value)
        self.dt = 1.0 / rate

        self.observer = TerrainObserver(
            freefall_accel_thresh=float(self.get_parameter('freefall_accel_thresh').value),
            impact_accel_thresh=float(self.get_parameter('impact_accel_thresh').value),
            ramp_pitch_thresh=float(self.get_parameter('ramp_pitch_thresh').value),
        )

        # Internal sensor states
        self.accel_x: float = 0.0
        self.accel_z: float = 9.81
        self.accel_mag: float = 9.81
        self.pitch: float = 0.0
        self.pitch_rate: float = 0.0
        self.v_meas: float = 0.0
        self.eff_height: float = 0.28
        self.z_vel: float = 0.0
        self.sim_time: float = 0.0
        self.forward_clearance: float = 12.0

        # Subscriptions
        self.sub_imu = self.create_subscription(
            Imu, '/imu/data', self._imu_callback, 10
        )
        self.sub_joints = self.create_subscription(
            JointState, '/joint_states', self._joint_callback, 10
        )
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10
        )

        # Publications
        self.pub_flight_state = self.create_publisher(
            Int32, '/balancebot/flight_state', 10
        )
        self.pub_terrain_slope = self.create_publisher(
            Float64, '/balancebot/terrain_slope', 10
        )
        self.pub_forward_clearance = self.create_publisher(
            Float64, '/balancebot/forward_clearance', 10
        )

        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info('Terrain Observer Node active (50 Hz).')

    def _scan_callback(self, msg: LaserScan) -> None:
        if msg.ranges:
            mid = len(msg.ranges) // 2
            span = max(1, int(len(msg.ranges) * (30.0 / 360.0) / 2))
            center_ranges = [r for r in msg.ranges[mid - span:mid + span] if msg.range_min <= r <= msg.range_max]
            if center_ranges:
                self.forward_clearance = float(min(center_ranges))

    def _imu_callback(self, msg: Imu) -> None:
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        self.accel_x = float(ax)
        self.accel_z = float(az)
        self.accel_mag = float(math.sqrt(ax**2 + ay**2 + az**2))
        self.pitch_rate = float(msg.angular_velocity.y)

        q = msg.orientation
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        self.pitch = float(math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp))

    def _joint_callback(self, msg: JointState) -> None:
        name_pos = dict(zip(msg.name, msg.position))
        name_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

        # 2-DOF sagittal leg height estimation (l1=0.20, l2=0.20)
        qlh = name_pos.get('left_hip_joint', 0.0)
        qlk = name_pos.get('left_knee_joint', 0.0)
        qrh = name_pos.get('right_hip_joint', 0.0)
        qrk = name_pos.get('right_knee_joint', 0.0)

        zl = -(0.20 * math.cos(qlh) + 0.20 * math.cos(qlh + qlk))
        zr = -(0.20 * math.cos(qrh) + 0.20 * math.cos(qrh + qrk))
        prev_h = self.eff_height
        self.eff_height = 0.5 * (-zl - zr)
        self.z_vel = (self.eff_height - prev_h) / max(self.dt, 1e-4)

        # Forward velocity from wheel velocities (r=0.10m)
        vl = name_vel.get('left_wheel_joint', 0.0) * 0.10
        vr = name_vel.get('right_wheel_joint', 0.0) * 0.10
        if msg.velocity:
            self.v_meas = 0.5 * (vl + vr)

    def _control_loop(self) -> None:
        self.sim_time += self.dt

        state = self.observer.update(
            t=self.sim_time,
            accel_mag=self.accel_mag,
            wheel_contact=True,
            pitch=self.pitch,
            z_vel=self.z_vel,
            dt=self.dt,
            pitch_rate=self.pitch_rate,
            v_meas=self.v_meas,
            ax=self.accel_x,
            az=self.accel_z,
        )

        state_msg = Int32()
        state_msg.data = int(state)
        self.pub_flight_state.publish(state_msg)

        slope_msg = Float64()
        slope_msg.data = float(self.observer.terrain_slope)
        self.pub_terrain_slope.publish(slope_msg)

        clearance_msg = Float64()
        clearance_msg.data = float(self.forward_clearance)
        self.pub_forward_clearance.publish(clearance_msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TerrainObserverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
