"""BalanceBot Evaluator Harness & Physics Simulation Oracle.

Provides the reference physical and kinematic models, Gain-Scheduled LQR-I balance controller,
active articulated leg kinematics, terrain models, 5-state flight/landing FSM, sensor simulation,
and quantitative metrics computation (J_v, J_theta, T_s, clearance, recovery).
"""

from __future__ import annotations

import dataclasses
import enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg


@dataclasses.dataclass
class RobotParams:
    """Robot physical and kinematic parameters."""
    m_b: float = 10.0            # Torso mass (kg)
    M_w: float = 1.2             # Wheel mass each (kg)
    R: float = 0.10              # Wheel radius (m)
    I_b: float = 0.135           # Torso pitch inertia (kg*m^2)
    I_w: float = 0.006           # Wheel polar inertia (kg*m^2)
    L1: float = 0.20             # Thigh length (m)
    L2: float = 0.20             # Shank length (m)
    track_width: float = 0.45    # Wheel track width (m)
    tau_max: float = 15.0        # Max wheel torque (N*m)
    g: float = 9.81              # Gravity (m/s^2)
    height_min: float = 0.18     # Minimum body height (m)
    height_max: float = 0.38     # Maximum body height (m)
    offset_min: float = -0.08    # Max rearward wheel offset (m)
    offset_max: float = 0.08     # Max forward wheel offset (m)
    hip_limit_rad: float = math.radians(60.0)
    knee_min_rad: float = math.radians(0.0)
    knee_max_rad: float = math.radians(120.0)

    @property
    def total_mass(self) -> float:
        return self.m_b + 2.0 * self.M_w

    @property
    def m_total_effective(self) -> float:
        return self.m_b + 2.0 * self.M_w + (2.0 * self.I_w) / (self.R**2)


class TerrainType(enum.Enum):
    FLAT = "flat"
    BUMPS = "bumps"
    ROUGH = "rough"
    SLOPE = "slope"
    RAMP = "ramp"


class FlightState(enum.IntEnum):
    GROUND_BALANCE = 0
    RAMP_ASCENT = 1
    AIRBORNE = 2
    TOUCHDOWN_ABSORPTION = 3
    BALANCE_RECOVERY = 4


class SkateparkEnvironment:
    """Skatepark multi-terrain world elevation and slope profile."""

    def __init__(self) -> None:
        # Zone boundaries along x axis (meters)
        self.flat_zone = (-20.0, 5.0)
        self.bumps_zone = (5.0, 10.0)
        self.rough_zone = (10.0, 15.0)
        self.slope_zone = (15.0, 20.0)
        self.ramp_zone = (20.0, 23.0)

        # Bump and ramp parameters
        self.bump_amplitude = 0.03       # 3 cm bumps
        self.bump_wavelength = 0.8       # 0.8 m wavelength
        self.slope_angle = math.radians(12.0)  # 12 deg slope
        self.ramp_incline = math.radians(20.0) # 20 deg ramp
        self.ramp_height = 0.35          # 35 cm lip height
        self.friction_coeff = 1.0
        self.restitution = 0.04

    def get_zone(self, x: float) -> TerrainType:
        if x < self.bumps_zone[0]:
            return TerrainType.FLAT
        if x < self.rough_zone[0]:
            return TerrainType.BUMPS
        if x < self.slope_zone[0]:
            return TerrainType.ROUGH
        if x < self.ramp_zone[0]:
            return TerrainType.SLOPE
        return TerrainType.RAMP

    def elevation(self, x: float, y: float = 0.0) -> float:
        if x < 5.0:
            return 0.0
        if x < 10.0:
            # Sinusoidal bumps
            return self.bump_amplitude * math.sin(2.0 * math.pi * (x - 5.0) / self.bump_wavelength)
        if x < 15.0:
            # Rough irregular blocks
            dx = x - 10.0
            return 0.025 * math.sin(7.0 * dx) + 0.015 * math.cos(13.0 * dx)
        if x < 20.0:
            # Planar slope
            return (x - 15.0) * math.tan(self.slope_angle)
        if x <= 23.0:
            # Ramp incline up to 0.35m
            base_z = 5.0 * math.tan(self.slope_angle)
            return base_z + (x - 20.0) * (self.ramp_height / 3.0)
        # Drop after ramp lip
        return 0.0

    def slope_angle_at(self, x: float) -> float:
        if x < 5.0:
            return 0.0
        if x < 10.0:
            k = 2.0 * math.pi / self.bump_wavelength
            dzdx = self.bump_amplitude * k * math.cos(k * (x - 5.0))
            return math.atan(dzdx)
        if x < 15.0:
            dx = x - 10.0
            dzdx = 0.025 * 7.0 * math.cos(7.0 * dx) - 0.015 * 13.0 * math.sin(13.0 * dx)
            return math.atan(dzdx)
        if x < 20.0:
            return self.slope_angle
        if x <= 23.0:
            return self.ramp_incline
        return 0.0


class LegKinematics:
    """Forward & inverse kinematics and virtual model impedance control for 2-DOF legs."""

    def __init__(self, params: Optional[RobotParams] = None) -> None:
        self.p = params or RobotParams()

    def forward_kinematics(self, q_h: float, q_k: float) -> Tuple[float, float]:
        """Compute relative wheel center position (x_rel, z_rel) given hip and knee angles."""
        x_rel = self.p.L1 * math.sin(q_h) + self.p.L2 * math.sin(q_h + q_k)
        z_rel = -(self.p.L1 * math.cos(q_h) + self.p.L2 * math.cos(q_h + q_k))
        return x_rel, z_rel

    def inverse_kinematics(self, x_d: float, z_d: float) -> Tuple[float, float]:
        """Compute (q_h, q_k) for desired relative foot position (x_d, z_d) where z_d < 0."""
        # Clamp requested height within physical limits
        height = -z_d
        height_clamped = np.clip(height, self.p.height_min, self.p.height_max)
        z_clamped = -height_clamped
        x_clamped = np.clip(x_d, self.p.offset_min, self.p.offset_max)

        dist_sq = x_clamped**2 + z_clamped**2
        d_val = (dist_sq - self.p.L1**2 - self.p.L2**2) / (2.0 * self.p.L1 * self.p.L2)
        d_val = float(np.clip(d_val, -0.98, 0.95))

        q_k = math.acos(d_val)
        q_h = math.atan2(x_clamped, -z_clamped) - math.atan2(
            self.p.L2 * math.sin(q_k), self.p.L1 + self.p.L2 * math.cos(q_k)
        )
        return q_h, q_k

    def jacobian(self, q_h: float, q_k: float) -> np.ndarray:
        """Leg Jacobian matrix mapping joint rates to Cartesian velocity."""
        j11 = self.p.L1 * math.cos(q_h) + self.p.L2 * math.cos(q_h + q_k)
        j12 = self.p.L2 * math.cos(q_h + q_k)
        j21 = self.p.L1 * math.sin(q_h) + self.p.L2 * math.sin(q_h + q_k)
        j22 = self.p.L2 * math.sin(q_h + q_k)
        return np.array([[j11, j12], [j21, j22]], dtype=float)

    def virtual_model_torque(
        self,
        q_h: float,
        q_k: float,
        target_z: float,
        current_z: float,
        z_vel: float,
        k_z: float = 1200.0,
        d_z: float = 80.0,
        f_gravity_ff: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute hip and knee joint torques from Cartesian virtual spring-damper."""
        if f_gravity_ff is None:
            f_gravity_ff = (self.p.total_mass * self.p.g) / 2.0
        f_z = k_z * (target_z - current_z) - d_z * z_vel + f_gravity_ff
        f_cartesian = np.array([0.0, f_z], dtype=float)
        j = self.jacobian(q_h, q_k)
        tau = j.T @ f_cartesian
        return float(tau[0]), float(tau[1])

    def equilibrium_pitch(self, offset_x: float, effective_length: float) -> float:
        """CoM equilibrium pitch angle corresponding to fore/aft wheel offset."""
        eff_l = max(effective_length, self.p.height_min)
        ratio = offset_x / eff_l
        ratio = float(np.clip(ratio, -0.6, 0.6))
        return math.asin(ratio)


class LQRBalanceController:
    """Gain-Scheduled State-Feedback LQR with Integral Action (LQR-I)."""

    def __init__(self, params: Optional[RobotParams] = None) -> None:
        self.p = params or RobotParams()
        self.gain_grid: Dict[float, np.ndarray] = {}
        self._build_gain_grid()

    def _build_gain_grid(self) -> None:
        """Precompute LQR gains across effective pendulum length grid [0.18, 0.38] m."""
        q_mat = np.diag([0.0, 15.0, 80.0, 8.0, 5.0])
        r_mat = np.array([[1.0]])

        for length in np.linspace(0.18, 0.38, 21):
            length = round(length, 3)
            m_total = self.p.m_total_effective
            delta = m_total * (self.p.I_b + self.p.m_b * length**2) - (self.p.m_b * length)**2

            a_mat = np.array([
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -(self.p.m_b * length)**2 * self.p.g / delta, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, m_total * self.p.m_b * self.p.g * length / delta, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ])
            b_mat = np.array([
                [0.0],
                [(2.0 * (self.p.I_b + self.p.m_b * length**2) / self.p.R + self.p.m_b * length) / delta],
                [0.0],
                [-(2.0 * self.p.m_b * length / self.p.R + m_total) / delta],
                [0.0],
            ])

            p_sol = scipy.linalg.solve_continuous_are(a_mat, b_mat, q_mat, r_mat)
            k_gains = (np.linalg.inv(r_mat) @ b_mat.T @ p_sol).flatten()
            self.gain_grid[length] = k_gains

    def get_gains(self, effective_length: float) -> np.ndarray:
        """Interpolate LQR gains for instantaneous effective pendulum length."""
        l_clamped = float(np.clip(effective_length, 0.18, 0.38))
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
        """Compute mean drive wheel torque and anti-windup status."""
        gains = self.get_gains(effective_length)
        e_pos = pos_meas - pos_ref if pos_ref != 0.0 else 0.0
        e_vel = v_meas - v_ref
        e_theta = theta_meas - theta_ref

        state = np.array([e_pos, e_vel, e_theta, theta_dot, e_integral])
        tau_raw = -float(np.dot(gains, state))

        tau_clamped = float(np.clip(tau_raw, -self.p.tau_max, self.p.tau_max))
        saturated = bool(abs(tau_raw) >= self.p.tau_max)
        return tau_clamped, saturated


class FlightFSM:
    """5-State flight and landing shock absorption finite state machine."""

    def __init__(self) -> None:
        self.state: FlightState = FlightState.GROUND_BALANCE
        self.freefall_timer: float = 0.0
        self.recovery_timer: float = 0.0
        self.touchdown_time: Optional[float] = None
        self.airborne_time: Optional[float] = None

    def update(
        self,
        t: float,
        accel_mag: float,
        wheel_contact: bool,
        pitch: float,
        z_vel: float,
        dt: float,
    ) -> FlightState:
        """Update FSM transitions based on IMU specific force and contact state."""
        if self.state == FlightState.GROUND_BALANCE:
            if not wheel_contact or accel_mag < 2.5:
                self.freefall_timer += dt
                if self.freefall_timer >= 0.030:  # 30 ms free-fall detection
                    self.state = FlightState.AIRBORNE
                    self.airborne_time = t
                    self.freefall_timer = 0.0
            else:
                self.freefall_timer = 0.0

        elif self.state == FlightState.RAMP_ASCENT:
            if not wheel_contact or accel_mag < 2.5:
                self.state = FlightState.AIRBORNE
                self.airborne_time = t

        elif self.state == FlightState.AIRBORNE:
            # Detect touchdown impact
            if wheel_contact or accel_mag > 15.0:
                self.state = FlightState.TOUCHDOWN_ABSORPTION
                self.touchdown_time = t
                self.recovery_timer = 0.0

        elif self.state == FlightState.TOUCHDOWN_ABSORPTION:
            # Impact absorbed when vertical displacement settles
            if abs(z_vel) < 0.05 or (self.touchdown_time and (t - self.touchdown_time) > 0.25):
                self.state = FlightState.BALANCE_RECOVERY
                self.recovery_timer = 0.0

        elif self.state == FlightState.BALANCE_RECOVERY:
            self.recovery_timer += dt
            if abs(pitch) < math.radians(5.0) and abs(z_vel) < 0.03:
                self.state = FlightState.GROUND_BALANCE

        return self.state


class SensorSuite:
    """Simulated IMU, 2D LiDAR, and RGB-D Depth Camera sensor models."""

    def __init__(self) -> None:
        self.imu_update_count: int = 0
        self.lidar_update_count: int = 0
        self.camera_update_count: int = 0
        self.last_imu_time: float = 0.0
        self.last_lidar_time: float = 0.0
        self.last_camera_time: float = 0.0
        self.filtered_pitch: float = 0.0

    def step(
        self,
        t: float,
        true_pitch: float,
        true_pitch_rate: float,
        true_accel: np.ndarray,
        x_pos: float,
        environment: SkateparkEnvironment,
    ) -> Dict[str, Any]:
        """Update sensor streams and compute fusion state."""
        sensors: Dict[str, Any] = {}

        # IMU at 100 Hz
        if t - self.last_imu_time >= 0.0099:
            self.imu_update_count += 1
            self.last_imu_time = t
            accel_meas = true_accel + np.random.normal(0, 0.02, size=3)
            gyro_meas = true_pitch_rate + np.random.normal(0, 0.005)
            # Complementary filter pitch fusion
            pitch_acc = math.atan2(accel_meas[0], math.sqrt(accel_meas[1]**2 + accel_meas[2]**2))
            alpha = 0.98
            self.filtered_pitch = alpha * (self.filtered_pitch + gyro_meas * 0.01) + (1.0 - alpha) * pitch_acc
            sensors["imu"] = {
                "accel": accel_meas,
                "gyro": gyro_meas,
                "pitch": self.filtered_pitch,
                "frequency": 100.0,
            }

        # LiDAR at 10 Hz
        if t - self.last_lidar_time >= 0.099:
            self.lidar_update_count += 1
            self.last_lidar_time = t
            angles = np.linspace(-math.pi / 2, math.pi / 2, 90)
            ranges = []
            for ang in angles:
                look_x = x_pos + 4.0 * math.cos(ang)
                elev = environment.elevation(look_x)
                dist = math.sqrt((look_x - x_pos)**2 + elev**2)
                ranges.append(float(np.clip(dist, 0.1, 12.0)))
            sensors["lidar"] = {
                "ranges": np.array(ranges),
                "frequency": 10.0,
            }

        # Depth Camera at 15 Hz
        if t - self.last_camera_time >= 0.066:
            self.camera_update_count += 1
            self.last_camera_time = t
            depth_map = np.ones((48, 64), dtype=np.float32) * 5.0
            sensors["depth_camera"] = {
                "depth_map": depth_map,
                "resolution": (640, 480),
                "frequency": 15.0,
            }

        return sensors


class BalanceBotSimulator:
    """Headless 6-DOF dynamic simulator with active legs, terrain contact, and landing shock absorption."""

    def __init__(self, params: Optional[RobotParams] = None) -> None:
        self.p = params or RobotParams()
        self.env = SkateparkEnvironment()
        self.kin = LegKinematics(self.p)
        self.lqr = LQRBalanceController(self.p)
        self.fsm = FlightFSM()
        self.sensors = SensorSuite()

        # States: x, v, theta, theta_dot, yaw, yaw_rate, z_body, z_dot
        self.reset()

    def reset(
        self,
        x0: float = 0.0,
        v0: float = 0.0,
        theta0: float = 0.0,
        height0: float = 0.28,
    ) -> None:
        """Reset simulation states to initial conditions."""
        self.x = x0
        self.v = v0
        self.theta = theta0
        self.theta_dot = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.effective_height = height0
        self.z_vel = 0.0
        self.wheel_offset = 0.0
        self.e_integral = 0.0
        self.t = 0.0
        self.wheel_contact = True
        self.leg_mode = "auto"
        self.target_height = height0
        self.target_offset = 0.0
        self.q_h, self.q_k = self.kin.inverse_kinematics(0.0, -height0)
        self.history: List[Dict[str, float]] = []

    def set_leg_mode(self, mode: str, target_height: Optional[float] = None, target_offset: Optional[float] = None) -> None:
        """Switch leg regulation mode: 'auto' or 'manual'."""
        assert mode in ("auto", "manual"), f"Invalid mode: {mode}"
        self.leg_mode = mode
        if target_height is not None:
            self.target_height = float(np.clip(target_height, self.p.height_min, self.p.height_max))
        if target_offset is not None:
            self.target_offset = float(np.clip(target_offset, self.p.offset_min, self.p.offset_max))

    def step(
        self,
        dt: float = 0.001,
        v_ref: float = 0.0,
        omega_ref: float = 0.0,
        ext_force: float = 0.0,
        ext_torque: float = 0.0,
    ) -> Dict[str, float]:
        """Perform one numerical physics simulation step (dt = 0.001 s nominal)."""
        # 1. Kinematics & leg target regulation
        if self.leg_mode == "manual":
            # Slew rate limit toward target
            dz = np.clip(self.target_height - self.effective_height, -0.20 * dt, 0.20 * dt)
            dx = np.clip(self.target_offset - self.wheel_offset, -0.15 * dt, 0.15 * dt)
            self.effective_height += dz
            self.wheel_offset += dx
        else:
            # Auto mode: adjust height for terrain absorption
            slope = self.env.slope_angle_at(self.x)
            if self.env.get_zone(self.x) == TerrainType.BUMPS:
                desired_h = 0.28 - 0.5 * self.env.elevation(self.x)
            elif self.fsm.state == FlightState.RAMP_ASCENT:
                desired_h = 0.22
            elif self.fsm.state == FlightState.AIRBORNE:
                desired_h = 0.36
            elif self.fsm.state == FlightState.TOUCHDOWN_ABSORPTION:
                desired_h = 0.20
            else:
                desired_h = 0.28
            self.effective_height += np.clip(desired_h - self.effective_height, -0.5 * dt, 0.5 * dt)

        self.q_h, self.q_k = self.kin.inverse_kinematics(self.wheel_offset, -self.effective_height)

        # 2. Dynamic equilibrium shift
        theta_eq = self.kin.equilibrium_pitch(self.wheel_offset, self.effective_height)
        slope = self.env.slope_angle_at(self.x)
        total_theta_ref = theta_eq + slope

        # 3. Ground contact and vertical dynamics
        ground_z = self.env.elevation(self.x)
        chassis_bottom_z = ground_z + self.effective_height * math.cos(self.theta) - 0.10
        chassis_clearance = max(0.0, chassis_bottom_z - ground_z)

        # Check launch off ramp
        if self.x > 23.0 and ground_z == 0.0 and self.effective_height > 0.10 and self.v > 1.2:
            self.wheel_contact = False

        # 4. FSM state update
        accel_vec = np.array([
            self.v * math.cos(self.theta),
            0.0,
            self.p.g if not self.wheel_contact else 0.0,
        ])
        accel_mag = float(np.linalg.norm(accel_vec))
        fsm_state = self.fsm.update(
            self.t,
            accel_mag,
            self.wheel_contact,
            self.theta,
            self.z_vel,
            dt,
        )

        # 5. Wheel Torque Calculation
        if fsm_state == FlightState.AIRBORNE:
            # Suppress wheel balance torque in mid-air
            tau_w = 0.0
            saturated = False
        else:
            tau_w, saturated = self.lqr.compute_torque(
                v_meas=self.v,
                v_ref=v_ref,
                theta_meas=self.theta,
                theta_ref=total_theta_ref,
                theta_dot=self.theta_dot,
                e_integral=self.e_integral,
                effective_length=self.effective_height,
            )
            if not saturated:
                self.e_integral += (self.v - v_ref) * dt

        # Yaw steering torque
        delta_tau = 8.0 * (omega_ref - self.yaw_rate)
        tau_l = tau_w + 0.5 * delta_tau
        tau_r = tau_w - 0.5 * delta_tau

        # 6. Coupled WIPM Equations of Motion
        l = self.effective_height
        m_total = self.p.m_total_effective

        if not self.wheel_contact:
            # Ballistic free-flight in air: no ground rolling pivot, gravity acts at CoM
            accel_p = 0.0
            accel_theta = -1.0 * self.theta_dot
        else:
            det_m = m_total * (self.p.I_b + self.p.m_b * l**2) - (self.p.m_b * l * math.cos(self.theta))**2
            det_m = max(det_m, 1e-4)

            gravity_rolling_force = -self.p.total_mass * self.p.g * math.sin(slope)
            c_p = (
                self.p.m_b * l * (self.theta_dot**2) * math.sin(self.theta)
                + (2.0 / self.p.R) * tau_w
                + ext_force
                + gravity_rolling_force
                - 0.5 * self.v
            )
            c_theta = (
                self.p.m_b * self.p.g * l * math.sin(self.theta)
                - tau_w
                + ext_torque
                - 0.05 * self.theta_dot
            )

            inv11 = (self.p.I_b + self.p.m_b * l**2) / det_m
            inv12 = -(self.p.m_b * l * math.cos(self.theta)) / det_m
            inv21 = inv12
            inv22 = m_total / det_m

            accel_p = inv11 * c_p + inv12 * c_theta
            accel_theta = inv21 * c_p + inv22 * c_theta

        # Integrate longitudinal and rotational motion (Euler/Symplectic step)
        self.v += accel_p * dt
        self.x += self.v * dt
        self.theta_dot += accel_theta * dt
        self.theta += self.theta_dot * dt

        # Integrate yaw motion
        yaw_accel = delta_tau / (self.p.total_mass * (self.p.track_width / 2.0)**2)
        self.yaw_rate += yaw_accel * dt
        self.yaw += self.yaw_rate * dt

        # Check for landing re-contact if in air
        if not self.wheel_contact and self.x > 23.5:
            self.wheel_contact = True

        self.t += dt

        # Sensor updates
        self.sensors.step(
            self.t,
            self.theta,
            self.theta_dot,
            accel_vec,
            self.x,
            self.env,
        )

        record = {
            "time": self.t,
            "x": self.x,
            "v": self.v,
            "v_ref": v_ref,
            "theta": self.theta,
            "theta_ref": total_theta_ref,
            "theta_dot": self.theta_dot,
            "yaw": self.yaw,
            "yaw_rate": self.yaw_rate,
            "tau_w": tau_w,
            "height": self.effective_height,
            "offset": self.wheel_offset,
            "clearance": chassis_clearance,
            "fsm_state": int(fsm_state),
        }
        self.history.append(record)
        return record


class MetricsCalculator:
    """Quantitative performance metric calculator."""

    @staticmethod
    def calculate_integrated_errors(
        history: List[Dict[str, float]],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Compute integrated absolute velocity error J_v and pitch error J_theta."""
        j_v = 0.0
        j_theta = 0.0
        t_prev = None

        for rec in history:
            t = rec["time"]
            if t < t_start:
                continue
            if t_end is not None and t > t_end:
                break
            if t_prev is None:
                t_prev = t
                continue
            dt = t - t_prev
            t_prev = t
            j_v += abs(rec["v"] - rec["v_ref"]) * dt
            j_theta += abs(rec["theta"] - rec["theta_ref"]) * dt

        return j_v, j_theta

    @staticmethod
    def calculate_settling_time(
        history: List[Dict[str, float]],
        v_target: float,
        theta_target: float = 0.0,
        v_band: float = 0.10,
        theta_band_rad: float = math.radians(2.0),
    ) -> float:
        """Compute settling time T_s where state stays permanently within error tolerance band."""
        if not history:
            return 0.0

        t_settled = history[-1]["time"]
        for i in range(len(history) - 1, -1, -1):
            rec = history[i]
            v_err = abs(rec["v"] - v_target)
            theta_err = abs(rec["theta"] - theta_target)
            if v_err > v_band or theta_err > theta_band_rad:
                t_settled = history[i]["time"]
                break

        t_start = history[0]["time"]
        return max(0.0, t_settled - t_start)

    @staticmethod
    def verify_quantitative_criteria(
        history: List[Dict[str, float]],
        max_steady_pitch_deg: float = 5.0,
        max_j_v: float = 1.5,
        max_j_theta: float = 0.15,
        max_t_s: float = 2.0,
        min_clearance: float = 0.0,
        max_recovery_time: float = 2.5,
    ) -> Dict[str, Any]:
        """Verify the 5 authoritative quantitative acceptance criteria."""
        j_v, j_theta = MetricsCalculator.calculate_integrated_errors(history)

        # Steady-state pitch check (last 30% of run)
        n = len(history)
        steady_history = history[int(0.7 * n):] if n > 0 else history
        steady_pitches = [abs(math.degrees(r["theta"])) for r in steady_history]
        max_pitch_deg = max(steady_pitches) if steady_pitches else 0.0

        # Settling time
        v_ref = history[-1]["v_ref"] if history else 0.0
        theta_ref = history[-1]["theta_ref"] if history else 0.0
        t_s = MetricsCalculator.calculate_settling_time(history, v_ref, theta_ref)

        # Minimum clearance
        min_clear = min(r["clearance"] for r in history) if history else 0.0

        # Landing recovery
        touchdown_idx = next((i for i, r in enumerate(history) if r["fsm_state"] == FlightState.TOUCHDOWN_ABSORPTION), None)
        recovery_time = 0.0
        if touchdown_idx is not None:
            t_touch = history[touchdown_idx]["time"]
            for r in history[touchdown_idx:]:
                if abs(r["theta"]) < math.radians(5.0):
                    recovery_time = r["time"] - t_touch
                    break

        results = {
            "steady_pitch_deg": float(max_pitch_deg),
            "steady_pitch_pass": bool(max_pitch_deg < max_steady_pitch_deg),
            "j_v": float(j_v),
            "j_v_pass": bool(j_v <= max_j_v),
            "j_theta": float(j_theta),
            "j_theta_pass": bool(j_theta <= max_j_theta),
            "settling_time": float(t_s),
            "settling_time_pass": bool(t_s < max_t_s),
            "min_clearance": float(min_clear),
            "clearance_pass": bool(min_clear > min_clearance),
            "recovery_time": float(recovery_time),
            "recovery_pass": bool(recovery_time <= max_recovery_time),
            "all_passed": bool(
                max_pitch_deg < max_steady_pitch_deg
                and j_v <= max_j_v
                and j_theta <= max_j_theta
                and t_s < max_t_s
                and min_clear > min_clearance
                and recovery_time <= max_recovery_time
            ),
        }
        return results
