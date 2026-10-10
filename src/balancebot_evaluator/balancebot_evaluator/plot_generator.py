"""Publication-ready Plot and Video Demonstration Generator for BalanceBot.

Generates:
1. Velocity tracking plot with settling corridor.
2. Pitch angle tracking plot with +/-5 deg limits.
3. Phase portrait (theta vs theta_dot) confirming stability.
4. Chassis clearance and FSM state progression.
5. Consolidated 2x2 evaluation dashboard.
6. Demonstration MP4 video with offscreen MuJoCo renderer and live telemetry HUD.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FSM_NAMES = {
    0: "GROUND_BALANCE",
    1: "RAMP_ASCENT",
    2: "AIRBORNE",
    3: "TOUCHDOWN_ABSORPTION",
    4: "BALANCE_RECOVERY",
}

FSM_COLORS = {
    0: "#2ca02c",  # Green
    1: "#ff7f0e",  # Orange
    2: "#d62728",  # Red
    3: "#9467bd",  # Purple
    4: "#1f77b4",  # Blue
}


class PlotGenerator:
    """Generates verification plots and video demonstrations."""

    @staticmethod
    def generate_plots(
        history: List[Dict[str, float]],
        metrics: Dict[str, Any],
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate individual analysis plots and evaluation dashboard."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        if not history:
            return paths

        times = np.array([r["time"] for r in history])
        vels = np.array([r["v"] for r in history])
        v_refs = np.array([r["v_ref"] for r in history])
        thetas_deg = np.array([math.degrees(r["theta"]) for r in history])
        theta_refs_deg = np.array([math.degrees(r.get("theta_ref", 0.0)) for r in history])
        theta_dots = np.array([r.get("theta_dot", 0.0) for r in history])
        clearances = np.array([r.get("clearance", 0.1) for r in history])
        states = np.array([r.get("fsm_state", 0) for r in history])

        # -------------------------------------------------------------
        # 1. Velocity Tracking Plot
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.plot(times, v_refs, "k--", label=r"Reference $v_{ref}$", linewidth=1.5)
        ax.plot(times, vels, "b-", label=r"Measured $v(t)$", linewidth=1.8)
        ax.fill_between(
            times,
            v_refs - 0.10,
            v_refs + 0.10,
            color="b",
            alpha=0.15,
            label=r"$\pm 0.10$ m/s Tolerance Band",
        )
        jv = metrics.get("j_v", 0.0)
        ts = metrics.get("settling_time", 0.0)
        ax.set_title(
            f"Velocity Tracking ($J_v = {jv:.3f}$ m/s·s, $T_s = {ts:.2f}$ s)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linear Velocity (m/s)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        vel_path = os.path.join(output_dir, "velocity_tracking.png")
        fig.tight_layout()
        fig.savefig(vel_path)
        plt.close(fig)
        paths["velocity_tracking"] = vel_path

        # -------------------------------------------------------------
        # 2. Pitch Angle Tracking Plot
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        ax.plot(times, theta_refs_deg, "k--", label=r"Reference $\theta_{ref}$", linewidth=1.5)
        ax.plot(times, thetas_deg, "g-", label=r"Torso Pitch $\theta(t)$", linewidth=1.8)
        ax.axhline(5.0, color="r", linestyle="--", alpha=0.8, label=r"Acceptance Limit ($\pm 5.0^\circ$)")
        ax.axhline(-5.0, color="r", linestyle="--", alpha=0.8)
        jth = metrics.get("j_theta", 0.0)
        ss_pitch = metrics.get("steady_pitch_deg", 0.0)
        ax.set_title(
            f"Pitch Angle Tracking ($J_\\theta = {jth:.4f}$ rad·s, Steady Max $= {ss_pitch:.2f}^\\circ$)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torso Pitch (deg)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        pitch_path = os.path.join(output_dir, "pitch_tracking.png")
        fig.tight_layout()
        fig.savefig(pitch_path)
        plt.close(fig)
        paths["pitch_tracking"] = pitch_path

        # -------------------------------------------------------------
        # 3. Phase Portrait Plot (theta vs theta_dot)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(thetas_deg, theta_dots, "m-", linewidth=1.5, label="Trajectory")
        ax.plot(thetas_deg[0], theta_dots[0], "go", markersize=7, label="Initial State")
        ax.plot(thetas_deg[-1], theta_dots[-1], "rx", markersize=8, label="Terminal State")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title("Sagittal Pitch Phase Portrait", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Pitch Angle $\theta$ (deg)")
        ax.set_ylabel(r"Pitch Angular Velocity $\dot{\theta}$ (rad/s)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best")
        phase_path = os.path.join(output_dir, "phase_portrait.png")
        fig.tight_layout()
        fig.savefig(phase_path)
        plt.close(fig)
        paths["phase_portrait"] = phase_path

        # -------------------------------------------------------------
        # 4. Clearance & Flight FSM State Timeline
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=150)
        ax1.plot(times, clearances, "c-", linewidth=1.8, label=r"Chassis Clearance $z(t)$")
        ax1.axhline(0.0, color="r", linestyle="--", linewidth=1.5, label="Ground Limit ($z=0$)")
        min_c = metrics.get("min_clearance", 0.0)
        ax1.set_title(f"Chassis Ground Clearance ($z_{{min}} = {min_c:.3f}$ m)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Clearance (m)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        ax2.step(times, states, "tab:orange", where="post", linewidth=1.8)
        ax2.set_yticks(list(FSM_NAMES.keys()))
        ax2.set_yticklabels([FSM_NAMES[k] for k in FSM_NAMES.keys()], fontsize=8)
        ax2.set_title("5-State Flight FSM Progression", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("FSM State")
        ax2.grid(True, linestyle=":", alpha=0.6)
        clearance_path = os.path.join(output_dir, "clearance_fsm.png")
        fig.tight_layout()
        fig.savefig(clearance_path)
        plt.close(fig)
        paths["clearance_fsm"] = clearance_path

        # -------------------------------------------------------------
        # 5. Consolidated 2x2 Evaluation Dashboard
        # -------------------------------------------------------------
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        # Panel (0, 0): Velocity Tracking
        axs[0, 0].plot(times, v_refs, "k--", label=r"$v_{ref}$")
        axs[0, 0].plot(times, vels, "b-", label=r"$v(t)$")
        axs[0, 0].fill_between(times, v_refs - 0.1, v_refs + 0.1, color="b", alpha=0.15)
        axs[0, 0].set_title(f"Velocity Tracking ($J_v={jv:.3f}$ m/s·s)", fontweight="bold")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Velocity (m/s)")
        axs[0, 0].grid(True, linestyle=":", alpha=0.6)
        axs[0, 0].legend()

        # Panel (0, 1): Pitch Tracking
        axs[0, 1].plot(times, theta_refs_deg, "k--", label=r"$\theta_{ref}$")
        axs[0, 1].plot(times, thetas_deg, "g-", label=r"$\theta(t)$")
        axs[0, 1].axhline(5.0, color="r", linestyle="--")
        axs[0, 1].axhline(-5.0, color="r", linestyle="--")
        axs[0, 1].set_title(f"Pitch Angle ($J_\\theta={jth:.4f}$ rad·s)", fontweight="bold")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Pitch (deg)")
        axs[0, 1].grid(True, linestyle=":", alpha=0.6)
        axs[0, 1].legend()

        # Panel (1, 0): Phase Portrait
        axs[1, 0].plot(thetas_deg, theta_dots, "m-", linewidth=1.2)
        axs[1, 0].set_title("Pitch Phase Portrait", fontweight="bold")
        axs[1, 0].set_xlabel(r"$\theta$ (deg)")
        axs[1, 0].set_ylabel(r"$\dot{\theta}$ (rad/s)")
        axs[1, 0].grid(True, linestyle=":", alpha=0.6)

        # Panel (1, 1): Clearance & Scorecard Summary
        axs[1, 1].plot(times, clearances, "c-", label="Clearance")
        axs[1, 1].axhline(0.0, color="r", linestyle="--", label="Limit")
        axs[1, 1].set_title(f"Chassis Clearance ($z_{{min}}={min_c:.3f}$ m)", fontweight="bold")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Clearance (m)")
        axs[1, 1].grid(True, linestyle=":", alpha=0.6)
        axs[1, 1].legend()

        fig.suptitle("RL-Robust-BalanceBot Comprehensive Evaluation Dashboard", fontsize=15, fontweight="bold")
        fig.tight_layout()
        dash_path = os.path.join(output_dir, "evaluation_dashboard.png")
        fig.savefig(dash_path)
        plt.close(fig)
        paths["evaluation_dashboard"] = dash_path

        return paths

    @staticmethod
    def render_video(
        history: List[Dict[str, float]],
        xml_path: str,
        output_mp4: str,
        fps: int = 30,
        camera_name: str = "tracking",
    ) -> str:
        """Render demonstration MP4 video with MuJoCo offscreen camera and HUD telemetry overlay."""
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
        if not history:
            return output_mp4

        import mujoco

        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        width, height = 640, 480
        renderer = mujoco.Renderer(m, height, width)

        cam = mujoco.MjvCamera()
        if camera_name == "tracking":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = m.body("base_link").id
            cam.distance = 2.8
            cam.elevation = -12.0
            cam.azimuth = 90.0
        elif camera_name == "front":
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot_front_camera")
        else:
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = m.body("base_link").id
            cam.distance = 3.5
            cam.elevation = -15.0
            cam.azimuth = 75.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_mp4, fourcc, float(fps), (width, height))

        total_sim_time = history[-1]["time"] - history[0]["time"]
        num_target_frames = max(1, int(total_sim_time * fps))
        sample_step = max(1, len(history) // num_target_frames)
        sampled_records = history[::sample_step]

        for rec in sampled_records:
            t = rec["time"]
            x = rec["x"]
            v = rec["v"]
            v_ref = rec.get("v_ref", 0.0)
            theta = rec["theta"]
            theta_deg = math.degrees(theta)
            h_eff = rec.get("height", 0.28)
            offset_x = rec.get("offset", 0.0)
            clearance = rec.get("clearance", 0.1)
            fsm_state = int(rec.get("fsm_state", 0))
            fsm_str = FSM_NAMES.get(fsm_state, "UNKNOWN")

            # Inverse kinematics for leg joints
            dist_sq = offset_x**2 + (-h_eff)**2
            d_val = (dist_sq - 0.20**2 - 0.20**2) / (2.0 * 0.20 * 0.20)
            d_val = float(np.clip(d_val, -0.98, 0.95))
            qk = math.acos(d_val)
            qh = math.atan2(offset_x, h_eff) - math.atan2(0.20 * math.sin(qk), 0.20 + 0.20 * math.cos(qk))

            # Update MuJoCo state
            d.qpos[0] = x
            d.qpos[1] = 0.0
            # Base z coordinate: ground elevation plus chassis clearance
            if x < 5.0:
                ground_z = 0.0
            elif x < 10.0:
                ground_z = 0.03 * math.sin(2.0 * math.pi * (x - 5.0) / 0.8)
            elif x < 15.0:
                ground_z = 0.025 * math.sin(7.0 * (x - 10.0)) + 0.015 * math.cos(13.0 * (x - 10.0))
            elif x < 20.0:
                ground_z = (x - 15.0) * math.tan(math.radians(12.0))
            elif x <= 23.0:
                ground_z = 5.0 * math.tan(math.radians(12.0)) + (x - 20.0) * (0.35 / 3.0)
            else:
                ground_z = 0.0
            d.qpos[2] = ground_z + max(clearance, 0.15)

            # Pitch quaternion
            d.qpos[3] = math.cos(theta / 2.0)
            d.qpos[4] = 0.0
            d.qpos[5] = math.sin(theta / 2.0)
            d.qpos[6] = 0.0

            # Joints
            d.qpos[7] = qh   # left hip
            d.qpos[8] = qk   # left knee
            d.qpos[9] = x / 0.10  # left wheel
            d.qpos[10] = qh  # right hip
            d.qpos[11] = qk  # right knee
            d.qpos[12] = x / 0.10 # right wheel

            mujoco.mj_forward(m, d)
            renderer.update_scene(d, cam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw Telemetry HUD Overlay
            # Top-left HUD box
            cv2.rectangle(bgr, (10, 10), (320, 125), (20, 20, 20), -1)
            cv2.rectangle(bgr, (10, 10), (320, 125), (100, 100, 100), 1)

            cv2.putText(bgr, f"TIME: {t:.2f} s | FSM: {fsm_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(bgr, f"VELOCITY: {v:.2f} m/s (ref: {v_ref:.2f})", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(bgr, f"PITCH: {theta_deg:+.1f} deg", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if abs(theta_deg) < 5.0 else (0, 0, 255), 1)
            cv2.putText(bgr, f"POS X: {x:.2f} m | Z: {d.qpos[2]:.2f} m", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(bgr, f"CLEARANCE: {clearance:.2f} m", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)

            # Brand badge bottom-right
            cv2.putText(bgr, "RL-Robust-BalanceBot Evaluator", (360, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            out.write(bgr)

        out.release()
        return output_mp4
