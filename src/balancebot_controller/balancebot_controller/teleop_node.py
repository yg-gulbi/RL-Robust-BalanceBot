#!/usr/bin/env python3
"""Keyboard Teleoperation & Deterministic Test Script Replay Node.

Publishes velocity commands to /cmd_vel (geometry_msgs/msg/Twist).
Supports both interactive keyboard control and deterministic automated replay.
"""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import tty
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TeleopNode(Node):
    """ROS 2 Node for interactive keyboard control and automated test script replay."""

    BUILTIN_PROFILES = {
        "step_velocity": [
            (1.0, 0.0, 0.0),
            (3.0, 0.8, 0.0),
            (2.0, 0.0, 0.0),
        ],
        "turn": [
            (1.0, 0.0, 0.0),
            (2.0, 0.5, 0.5),
            (2.0, 0.0, 0.0),
        ],
        "ramp": [
            (0.5, 0.0, 0.0),
            (2.0, 1.0, 0.0),  # ramps gradually
            (1.5, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        "multistage": [
            (1.0, 0.0, 0.0),
            (2.0, 0.8, 0.0),
            (1.5, 0.5, 0.4),
            (1.5, 0.0, 0.0),
        ],
    }

    def __init__(self) -> None:
        super().__init__("teleop_node")

        # Parameters
        self.declare_parameter("replay_mode", False)
        self.declare_parameter("replay_profile", "step_velocity")
        self.declare_parameter("linear_step", 0.1)
        self.declare_parameter("angular_step", 0.1)
        self.declare_parameter("max_linear", 2.0)
        self.declare_parameter("max_angular", 2.0)
        self.declare_parameter("publish_rate", 20.0)

        self.replay_mode = bool(self.get_parameter("replay_mode").value)
        self.profile_name = str(self.get_parameter("replay_profile").value)
        self.linear_step = float(self.get_parameter("linear_step").value)
        self.angular_step = float(self.get_parameter("angular_step").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        rate = float(self.get_parameter("publish_rate").value)
        self.dt = 1.0 / rate

        # State
        self.target_linear: float = 0.0
        self.target_angular: float = 0.0
        self.current_linear: float = 0.0
        self.current_angular: float = 0.0

        # Automated Replay state
        self.profile_stages: List[Tuple[float, float, float]] = []
        self.current_stage_idx: int = 0
        self.stage_elapsed: float = 0.0
        self.replay_finished: bool = False
        self._setup_replay_profile()

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # Terminal state for keyboard mode
        self.is_tty = sys.stdin.isatty()
        self.orig_term_settings = None
        if self.is_tty and not self.replay_mode:
            try:
                self.orig_term_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception as e:
                self.get_logger().warn(f"Could not set raw terminal: {e}")
                self.is_tty = False

        # Timer loop (20 Hz)
        self.timer = self.create_timer(self.dt, self._timer_callback)
        self.get_logger().info(
            f"Teleop Node initialized (replay_mode={self.replay_mode}, profile={self.profile_name})"
        )

    def _setup_replay_profile(self) -> None:
        """Parse or load selected automated replay profile."""
        if not self.replay_mode:
            return

        if self.profile_name in self.BUILTIN_PROFILES:
            self.profile_stages = list(self.BUILTIN_PROFILES[self.profile_name])
        elif os.path.isfile(self.profile_name):
            try:
                with open(self.profile_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.profile_stages = [(float(s[0]), float(s[1]), float(s[2])) for s in data]
            except Exception as e:
                self.get_logger().error(f"Failed to load replay profile file: {e}")
                self.profile_stages = list(self.BUILTIN_PROFILES["step_velocity"])
        else:
            self.get_logger().warn(f"Unknown profile '{self.profile_name}', using step_velocity")
            self.profile_stages = list(self.BUILTIN_PROFILES["step_velocity"])

    def _read_key(self) -> Optional[str]:
        """Non-blocking read of one keypress from stdin."""
        if not self.is_tty:
            return None
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.read(1)
        return None

    def _handle_keyboard(self) -> None:
        key = self._read_key()
        if key is None:
            return

        if key in ("w", "W"):
            self.target_linear = min(self.target_linear + self.linear_step, self.max_linear)
        elif key in ("s", "S"):
            self.target_linear = max(self.target_linear - self.linear_step, -self.max_linear)
        elif key in ("a", "A"):
            self.target_angular = min(self.target_angular + self.angular_step, self.max_angular)
        elif key in ("d", "D"):
            self.target_angular = max(self.target_angular - self.angular_step, -self.max_angular)
        elif key in (" ", "x", "X"):
            self.target_linear = 0.0
            self.target_angular = 0.0
        elif key in ("q", "Q"):
            self.get_logger().info("Quit key received. Stopping teleop.")
            self.target_linear = 0.0
            self.target_angular = 0.0

    def _handle_replay(self) -> None:
        if self.replay_finished:
            self.target_linear = 0.0
            self.target_angular = 0.0
            return

        if self.current_stage_idx >= len(self.profile_stages):
            self.replay_finished = True
            self.target_linear = 0.0
            self.target_angular = 0.0
            self.get_logger().info("Automated replay profile completed.")
            return

        duration, v_cmd, omega_cmd = self.profile_stages[self.current_stage_idx]
        self.target_linear = v_cmd
        self.target_angular = omega_cmd

        self.stage_elapsed += self.dt
        if self.stage_elapsed >= duration:
            self.current_stage_idx += 1
            self.stage_elapsed = 0.0
            if self.current_stage_idx < len(self.profile_stages):
                next_d, next_v, next_w = self.profile_stages[self.current_stage_idx]
                self.get_logger().info(
                    f"Replay advancing to stage {self.current_stage_idx + 1}/{len(self.profile_stages)}: "
                    f"v={next_v:.2f} m/s, omega={next_w:.2f} rad/s, duration={next_d:.2f}s"
                )

    def _timer_callback(self) -> None:
        if self.replay_mode:
            self._handle_replay()
        else:
            self._handle_keyboard()

        # Smooth slew rate limit towards targets
        linear_rate = 1.5 * self.dt
        angular_rate = 3.0 * self.dt
        dl = max(min(self.target_linear - self.current_linear, linear_rate), -linear_rate)
        da = max(min(self.target_angular - self.current_angular, angular_rate), -angular_rate)
        self.current_linear += dl
        self.current_angular += da

        twist = Twist()
        twist.linear.x = self.current_linear
        twist.angular.z = self.current_angular
        self.pub_cmd_vel.publish(twist)

    def destroy_node(self) -> bool:
        if self.orig_term_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_term_settings)
            except Exception:
                pass
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
