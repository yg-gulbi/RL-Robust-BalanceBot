#!/usr/bin/env python3
"""Interactive MuJoCo 3D Simulator & Dual-Mode Teleop for BalanceBot.

Controls are captured directly in the 3D Viewer Window AND via dedicated Remote Controller UI:
- [Arrow Up / W / I]    : Forward Drive (Continuous smooth cruise while tapped/held)
- [Arrow Down / S / ,]  : Backward Drive
- [Arrow Left / A / J]  : Turn Left
- [Arrow Right / D / L] : Turn Right
- [Space / K]           : Immediate Brake & Hold Position
- [P]                   : 40 N Push Disturbance Impulse
- [R]                   : Reset Pose
- [M]                   : Toggle Mode (Cruise Latch Mode vs Auto-Stop on Release)
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import tkinter as tk
from typing import Optional

import glfw
import mujoco
import mujoco.viewer
import numpy as np


class MuJoCoBalanceBotSim:
    """Real-time MuJoCo BalanceBot simulation engine with continuous teleoperation."""

    def __init__(self, xml_path: str) -> None:
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Model file not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Teleoperation reference states
        self.v_cmd: float = 0.0
        self.omega_cmd: float = 0.0
        self.pos_hold_x: float = 0.0
        self.pos_hold_y: float = 0.0
        self.is_holding: bool = True
        self.last_v_key_time: float = 0.0
        self.last_omega_key_time: float = 0.0

        # Control Modes:
        # 'cruise': Keys set target speed, maintains speed until Space/K or opposite key (teleop_twist style)
        # 'auto_stop': Keys maintain speed while actively tapped; stops after 1.0s of inactivity
        self.control_mode: str = "auto_stop"

        # Balance gains (tuned for WIPM)
        self.k_pitch: float = 42.0
        self.k_pitch_rate: float = 8.5
        self.k_vel: float = 10.0
        self.k_pos: float = 8.0
        self.k_steer: float = 2.8

        # Stance PD Gains
        self.kp_h: float = 300.0
        self.kd_h: float = 15.0
        self.kp_k: float = 400.0
        self.kd_k: float = 20.0
        self.gravity_ff_knee: float = -8.68

        self.running: bool = True
        self.status_msg: str = "Ready"
        self.reset()

    def reset(self) -> None:
        """Reset robot pose to nominal stance on flat ground."""
        self.data.qpos[0:3] = [0.0, 0.0, 0.428]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # root quaternion
        self.data.qpos[7] = -0.7954   # left hip (-45.6 deg)
        self.data.qpos[8] = 1.5908    # left knee (+91.1 deg)
        self.data.qpos[9] = 0.0       # left wheel
        self.data.qpos[10] = -0.7954  # right hip
        self.data.qpos[11] = 1.5908   # right knee
        self.data.qpos[12] = 0.0      # right wheel
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        self.pos_hold_x = 0.0
        self.pos_hold_y = 0.0
        self.is_holding = True
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.last_v_key_time = 0.0
        self.last_omega_key_time = 0.0
        self.status_msg = "Pose Reset (Holding Position)"
        mujoco.mj_forward(self.model, self.data)

    def handle_key(self, keycode: int) -> None:
        """Handle key press directly from MuJoCo 3D Viewer window, Tkinter, or terminal."""
        now = time.time()

        # Forward: Arrow Up (265), 'W' (87, 119), 'I' (73, 105)
        if keycode in (glfw.KEY_UP, ord('W'), ord('w'), ord('I'), ord('i')):
            self.v_cmd = min(1.2, max(0.3, self.v_cmd + 0.15))
            self.last_v_key_time = now
            self.is_holding = False
            self.status_msg = f"FORWARD speed: {self.v_cmd:.2f} m/s"
            print(f"\r>>> {self.status_msg}                     ", end="", flush=True)

        # Backward: Arrow Down (264), 'S' (83, 115), ',' (44)
        elif keycode in (glfw.KEY_DOWN, ord('S'), ord('s'), ord(',')):
            self.v_cmd = max(-0.8, min(-0.3, self.v_cmd - 0.15))
            self.last_v_key_time = now
            self.is_holding = False
            self.status_msg = f"BACKWARD speed: {self.v_cmd:.2f} m/s"
            print(f"\r>>> {self.status_msg}                    ", end="", flush=True)

        # Steer Left: Arrow Left (263), 'A' (65, 97), 'J' (74, 106)
        elif keycode in (glfw.KEY_LEFT, ord('A'), ord('a'), ord('J'), ord('j')):
            self.omega_cmd = min(1.5, max(0.4, self.omega_cmd + 0.25))
            self.last_omega_key_time = now
            self.status_msg = f"TURN LEFT rate: {self.omega_cmd:.2f} rad/s"
            print(f"\r>>> {self.status_msg}                   ", end="", flush=True)

        # Steer Right: Arrow Right (262), 'D' (68, 100), 'L' (76, 108)
        elif keycode in (glfw.KEY_RIGHT, ord('D'), ord('d'), ord('L'), ord('l')):
            self.omega_cmd = max(-1.5, min(-0.4, self.omega_cmd - 0.25))
            self.last_omega_key_time = now
            self.status_msg = f"TURN RIGHT rate: {self.omega_cmd:.2f} rad/s"
            print(f"\r>>> {self.status_msg}                  ", end="", flush=True)

        # Emergency Brake / Stop / Hold: Space (32), 'K' (75, 107)
        elif keycode in (glfw.KEY_SPACE, ord('K'), ord('k')):
            self.v_cmd = 0.0
            self.omega_cmd = 0.0
            self.last_v_key_time = 0.0
            self.last_omega_key_time = 0.0
            self.is_holding = True
            self.pos_hold_x = float(self.data.qpos[0])
            self.pos_hold_y = float(self.data.qpos[1])
            self.status_msg = f"STOPPED & HOLDING at ({self.pos_hold_x:.2f}, {self.pos_hold_y:.2f})"
            print(f"\r>>> {self.status_msg}      ", end="", flush=True)

        # Push disturbance: 'P' (80, 112)
        elif keycode in (ord('P'), ord('p')):
            self.data.qfrc_applied[0] = 40.0
            self.status_msg = "DISTURBANCE: 40 N Push applied!"
            print(f"\r>>> {self.status_msg}                     ", end="", flush=True)

        # Reset: 'R' (82, 114)
        elif keycode in (ord('R'), ord('r')):
            self.reset()
            print(f"\r>>> {self.status_msg}                     ", end="", flush=True)

        # Toggle Control Mode: 'M' (77, 109)
        elif keycode in (ord('M'), ord('m')):
            if self.control_mode == "auto_stop":
                self.control_mode = "cruise"
                self.status_msg = "MODE: Cruise Control (Maintains speed until Space/K)"
            else:
                self.control_mode = "auto_stop"
                self.status_msg = "MODE: Auto-Stop on Release (Brakes after key stops)"
            print(f"\r>>> {self.status_msg}                     ", end="", flush=True)

    def step_controller(self) -> None:
        """Compute closed-loop control torques and step simulation by 1 ms."""
        now = time.time()

        # In Auto-Stop Mode: If no key repeated within 0.8 seconds (generous typematic window), smoothly brake to stop
        if self.control_mode == "auto_stop":
            if now - self.last_v_key_time > 0.80:
                if abs(self.v_cmd) > 1e-3:
                    # Smoothly decelerate at 1.0 m/s^2
                    self.v_cmd = float(np.sign(self.v_cmd) * max(0.0, abs(self.v_cmd) - 1.0 * 0.001))
                else:
                    self.v_cmd = 0.0

            if now - self.last_omega_key_time > 0.80:
                if abs(self.omega_cmd) > 1e-3:
                    self.omega_cmd = float(np.sign(self.omega_cmd) * max(0.0, abs(self.omega_cmd) - 2.0 * 0.001))
                else:
                    self.omega_cmd = 0.0

        # 1. State extraction
        qw, qx, qy, qz = self.data.qpos[3:7]
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))
        pitch_rate = self.data.qvel[4]
        yaw_rate = self.data.qvel[5]

        qh_l, qk_l = self.data.qpos[7], self.data.qpos[8]
        qh_r, qk_r = self.data.qpos[10], self.data.qpos[11]
        dqh_l, dqk_l = self.data.qvel[6], self.data.qvel[7]
        dqh_r, dqk_r = self.data.qvel[9], self.data.qvel[10]

        # 2. Leg Stance Holding PD
        tau_hl = - self.kp_h * (qh_l - (-0.7954)) - self.kd_h * dqh_l
        tau_hr = - self.kp_h * (qh_r - (-0.7954)) - self.kd_h * dqh_r
        tau_kl = self.gravity_ff_knee - self.kp_k * (qk_l - 1.5908) - self.kd_k * dqk_l
        tau_kr = self.gravity_ff_knee - self.kp_k * (qk_r - 1.5908) - self.kd_k * dqk_r

        # 3. Forward Velocity in Heading Frame
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        v_forward = vx * math.cos(yaw) + vy * math.sin(yaw)

        # 4. Wheel Balance Control with Auto-Braking & Station-Keeping
        if abs(self.v_cmd) > 1e-3:
            # Driving mode: tracking commanded speed
            self.is_holding = False
            e_v = v_forward - self.v_cmd
            tau_w = (
                self.k_pitch * pitch
                + self.k_pitch_rate * pitch_rate
                + self.k_vel * e_v
            )
        else:
            # Idle / Stopped: Auto-brake and latch station-keeping position
            if not self.is_holding and abs(v_forward) < 0.08:
                self.is_holding = True
                self.pos_hold_x = float(self.data.qpos[0])
                self.pos_hold_y = float(self.data.qpos[1])

            if self.is_holding:
                # Firm position locking (Station-keeping)
                dx = self.data.qpos[0] - self.pos_hold_x
                dy = self.data.qpos[1] - self.pos_hold_y
                e_pos = dx * math.cos(yaw) + dy * math.sin(yaw)
                tau_w = (
                    self.k_pitch * pitch
                    + self.k_pitch_rate * pitch_rate
                    + 12.0 * v_forward
                    + self.k_pos * e_pos
                )
            else:
                # Active deceleration braking
                tau_w = (
                    self.k_pitch * pitch
                    + self.k_pitch_rate * pitch_rate
                    + 14.0 * v_forward
                )

        tau_w = float(np.clip(tau_w, -25.0, 25.0))

        # 5. Differential Steering
        delta_tau = self.k_steer * (self.omega_cmd - yaw_rate)
        delta_tau = float(np.clip(delta_tau, -10.0, 10.0))

        tau_l = float(np.clip(tau_w - delta_tau, -25.0, 25.0))
        tau_r = float(np.clip(tau_w + delta_tau, -25.0, 25.0))

        # 6. Apply to Actuators
        self.data.ctrl[0] = float(np.clip(tau_hl, -50.0, 50.0))
        self.data.ctrl[1] = float(np.clip(tau_hr, -50.0, 50.0))
        self.data.ctrl[2] = float(np.clip(tau_kl, -50.0, 50.0))
        self.data.ctrl[3] = float(np.clip(tau_kr, -50.0, 50.0))
        self.data.ctrl[4] = tau_l
        self.data.ctrl[5] = tau_r

        mujoco.mj_step(self.model, self.data)

    def run_gui(self) -> None:
        """Launch interactive MuJoCo 3D Viewer with direct window key events."""
        print("==================================================================")
        print("  MuJoCo RL-Robust-BalanceBot 3D Physics Simulator                ")
        print("  * Continuous Controls (Directly in 3D Window or Controller UI)  ")
        print("    [Arrow Up / W / I]     Drive Forward (Holds continuous speed) ")
        print("    [Arrow Down / S / ,]   Drive Backward                         ")
        print("    [Arrow Left / A / J]   Turn Left                              ")
        print("    [Arrow Right / D / L]  Turn Right                             ")
        print("    [Space / K]            Emergency Brake & Immediate Hold       ")
        print("    [M]                    Toggle Mode (Auto-Stop vs Cruise Latch)")
        print("    [P]                    40 N Push Disturbance Impulse          ")
        print("    [R]                    Reset Pose                             ")
        print("==================================================================")

        def viewer_key_callback(keycode: int) -> None:
            self.handle_key(keycode)

        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=viewer_key_callback
        ) as viewer:
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 90.0

            step_dt = self.model.opt.timestep

            while viewer.is_running() and self.running:
                step_start = time.time()

                self.step_controller()

                viewer.cam.lookat[0] = self.data.qpos[0]
                viewer.cam.lookat[1] = self.data.qpos[1]
                viewer.cam.lookat[2] = 0.35

                viewer.sync()

                elapsed = time.time() - step_start
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)


def launch_tk_controller(sim: MuJoCoBalanceBotSim) -> None:
    """Launch a tiny floating graphical remote controller window with physical key tracking."""
    try:
        root = tk.Tk()
        root.title("BalanceBot Controller")
        root.geometry("340x260+50+50")
        root.attributes("-topmost", True)
        root.configure(bg="#222831")

        title = tk.Label(root, text="🎮 BalanceBot Remote Control", font=("Arial", 12, "bold"), fg="#00ADB5", bg="#222831")
        title.pack(pady=8)

        status_lbl = tk.Label(root, text="Status: Ready", font=("Arial", 10), fg="#EEEEEE", bg="#222831")
        status_lbl.pack(pady=4)

        # Buttons grid
        frame = tk.Frame(root, bg="#222831")
        frame.pack(pady=8)

        btn_fwd = tk.Button(frame, text="▲ Forward (W/↑)", width=14, bg="#393E46", fg="white",
                            command=lambda: sim.handle_key(ord('w')))
        btn_fwd.grid(row=0, column=1, pady=3)

        btn_left = tk.Button(frame, text="◀ Left (A/←)", width=11, bg="#393E46", fg="white",
                             command=lambda: sim.handle_key(ord('a')))
        btn_left.grid(row=1, column=0, padx=3)

        btn_stop = tk.Button(frame, text="■ STOP (Space)", width=11, bg="#D63031", fg="white", font=("Arial", 9, "bold"),
                             command=lambda: sim.handle_key(glfw.KEY_SPACE))
        btn_stop.grid(row=1, column=1, padx=3)

        btn_right = tk.Button(frame, text="Right (D/→) ▶", width=11, bg="#393E46", fg="white",
                              command=lambda: sim.handle_key(ord('d')))
        btn_right.grid(row=1, column=2, padx=3)

        btn_bwd = tk.Button(frame, text="▼ Backward (S/↓)", width=14, bg="#393E46", fg="white",
                            command=lambda: sim.handle_key(ord('s')))
        btn_bwd.grid(row=2, column=1, pady=3)

        # Extra actions
        act_frame = tk.Frame(root, bg="#222831")
        act_frame.pack(pady=5)
        btn_push = tk.Button(act_frame, text="⚡ 40N Push (P)", width=14, bg="#E17055", fg="white",
                             command=lambda: sim.handle_key(ord('p')))
        btn_push.grid(row=0, column=0, padx=4)

        btn_reset = tk.Button(act_frame, text="↺ Reset (R)", width=12, bg="#636E72", fg="white",
                              command=lambda: sim.handle_key(ord('r')))
        btn_reset.grid(row=0, column=1, padx=4)

        # Physical Keyboard binding directly on Tkinter window
        def on_key(event):
            sim.handle_key(ord(event.char) if event.char else 0)

        root.bind("<Up>", lambda e: sim.handle_key(glfw.KEY_UP))
        root.bind("<Down>", lambda e: sim.handle_key(glfw.KEY_DOWN))
        root.bind("<Left>", lambda e: sim.handle_key(glfw.KEY_LEFT))
        root.bind("<Right>", lambda e: sim.handle_key(glfw.KEY_RIGHT))
        root.bind("<space>", lambda e: sim.handle_key(glfw.KEY_SPACE))
        root.bind("<Key>", on_key)

        def update_status():
            if sim.running:
                status_lbl.config(text=f"Speed: {sim.v_cmd:.2f} m/s | Turn: {sim.omega_cmd:.2f} rad/s\nMode: {sim.control_mode.upper()}")
                root.after(100, update_status)
            else:
                root.destroy()

        root.after(100, update_status)
        root.mainloop()
    except Exception as e:
        print(f"[Remote Controller UI Error]: {e}")


def terminal_keyboard_listener(sim: MuJoCoBalanceBotSim) -> None:
    """Terminal stdin fallback listener."""
    import select
    import termios
    import tty

    if not sys.stdin.isatty():
        return

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while sim.running:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                ch = sys.stdin.read(1)
                sim.handle_key(ord(ch))
                if ch in ('q', '\x03'):
                    sim.running = False
                    break
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_dir = os.path.abspath(script_dir)
    xml_path = os.path.join(
        ws_dir,
        "src", "balancebot_description", "worlds", "skatepark.xml"
    )

    if not os.path.exists(xml_path):
        xml_path = os.path.join(
            ws_dir,
            "install", "balancebot_description", "share", "balancebot_description", "worlds", "skatepark.xml"
        )

    sim = MuJoCoBalanceBotSim(xml_path)

    # 1. Start Terminal Keyboard Listener Thread
    t_thread = threading.Thread(target=terminal_keyboard_listener, args=(sim,), daemon=True)
    t_thread.start()

    # 2. Start Floating Graphical Remote Controller Window Thread
    gui_thread = threading.Thread(target=launch_tk_controller, args=(sim,), daemon=True)
    gui_thread.start()

    # 3. Main MuJoCo 3D Viewer Loop
    try:
        sim.run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        sim.running = False


if __name__ == "__main__":
    main()
