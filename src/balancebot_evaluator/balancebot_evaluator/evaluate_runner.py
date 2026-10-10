"""Automated Headless Evaluation Runner for BalanceBot.

Orchestrates multi-scenario physics verification, computes authoritative
quantitative metrics, exports scorecards (JSON, CSV, MD), generates
publication plots, and renders demonstration MP4 video with telemetry HUD.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .metrics_calculator import MetricsCalculator
from .plot_generator import PlotGenerator


def load_evaluator_harness() -> Tuple[Any, Any, Any]:
    """Dynamically load simulator and environment classes from evaluator harness."""
    try:
        from balancebot_evaluator.test.evaluator_harness import (
            BalanceBotSimulator,
            FlightState,
            SkateparkEnvironment,
        )
        return BalanceBotSimulator, SkateparkEnvironment, FlightState
    except ImportError:
        pass

    # Search for evaluator_harness.py in test directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(current_dir), "test", "evaluator_harness.py"),
        os.path.join(current_dir, "..", "test", "evaluator_harness.py"),
        os.path.abspath("src/balancebot_evaluator/test/evaluator_harness.py"),
    ]

    for cand in candidates:
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("evaluator_harness", cand)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["evaluator_harness"] = mod
                spec.loader.exec_module(mod)
                return mod.BalanceBotSimulator, mod.SkateparkEnvironment, mod.FlightState

    raise FileNotFoundError("Could not find evaluator_harness.py on search paths.")


class EvaluateRunner:
    """Headless simulation evaluation orchestrator."""

    def __init__(self, output_dir: str = "eval_output", render_video: bool = True) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.render_video_flag = render_video
        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.SimulatorClass, self.EnvClass, self.FlightStateClass = load_evaluator_harness()

    def run_cruise_scenario(self, duration_s: float = 5.0, v_target: float = 0.5) -> List[Dict[str, float]]:
        """Run flat ground balancing and cruise scenario."""
        sim = self.SimulatorClass()
        sim.reset(x0=0.0, v0=0.0, theta0=0.0, height0=0.28)
        num_steps = int(duration_s / 0.001)
        for _ in range(num_steps):
            sim.step(dt=0.001, v_ref=v_target)
        return sim.history

    def run_ramp_jump_scenario(self, duration_s: float = 4.0, v_target: float = 1.8) -> List[Dict[str, float]]:
        """Run ramp ascent, airborne launch, touchdown, and recovery scenario."""
        sim = self.SimulatorClass()
        sim.reset(x0=20.5, v0=v_target, theta0=0.0, height0=0.28)
        num_steps = int(duration_s / 0.001)
        for _ in range(num_steps):
            sim.step(dt=0.001, v_ref=v_target)
        return sim.history

    def run_evaluation(
        self, scenario: str = "all", fps: int = 30
    ) -> Dict[str, Any]:
        """Execute evaluation, compute metrics, and export all artifacts."""
        print("======================================================================")
        print("RL-Robust-BalanceBot Headless Evaluation Runner")
        print(f"Target Scenario: {scenario}")
        print(f"Output Directory: {self.output_dir}")
        print("======================================================================")

        t_start = time.time()
        combined_history: List[Dict[str, float]] = []

        if scenario in ("cruise", "flat"):
            history = self.run_cruise_scenario(duration_s=5.0, v_target=0.5)
            metrics = MetricsCalculator.verify_quantitative_criteria(history)
            combined_history = history
            eval_name = "Flat Ground Balance & Cruise"

        elif scenario in ("ramp_jump", "jump"):
            history = self.run_ramp_jump_scenario(duration_s=4.0, v_target=1.8)
            metrics = MetricsCalculator.verify_quantitative_criteria(history)
            combined_history = history
            eval_name = "Skatepark Ramp Jump & Landing Recovery"

        else:
            # Full comprehensive mission: evaluate both cruise and jump regimes
            cruise_hist = self.run_cruise_scenario(duration_s=5.0, v_target=0.5)
            jump_hist = self.run_ramp_jump_scenario(duration_s=4.0, v_target=1.8)

            cruise_metrics = MetricsCalculator.verify_quantitative_criteria(cruise_hist)
            jump_metrics = MetricsCalculator.verify_quantitative_criteria(jump_hist)

            # Combined authoritative scorecard:
            # Cruise regime validates tracking J_v, J_theta, settling time, steady pitch
            # Jump regime validates clearance and landing recovery
            min_clr = min(cruise_metrics["min_clearance"], jump_metrics["min_clearance"])
            rec_time = jump_metrics["recovery_time"]

            metrics = {
                "steady_pitch_deg": cruise_metrics["steady_pitch_deg"],
                "steady_pitch_pass": cruise_metrics["steady_pitch_pass"],
                "j_v": cruise_metrics["j_v"],
                "j_v_pass": cruise_metrics["j_v_pass"],
                "j_theta": cruise_metrics["j_theta"],
                "j_theta_pass": cruise_metrics["j_theta_pass"],
                "settling_time": cruise_metrics["settling_time"],
                "settling_time_pass": cruise_metrics["settling_time_pass"],
                "min_clearance": float(min_clr),
                "clearance_pass": bool(min_clr > 0.0),
                "recovery_time": float(rec_time),
                "recovery_pass": bool(rec_time <= 2.5),
                "all_passed": bool(
                    cruise_metrics["steady_pitch_pass"]
                    and cruise_metrics["j_v_pass"]
                    and cruise_metrics["j_theta_pass"]
                    and cruise_metrics["settling_time_pass"]
                    and (min_clr > 0.0)
                    and (rec_time <= 2.5)
                ),
            }
            combined_history = cruise_hist
            eval_name = "Comprehensive Skatepark Verification Suite"

        duration_total = time.time() - t_start

        # 1. Export Scorecard Files (JSON, CSV, Markdown)
        json_path = os.path.join(self.output_dir, "scorecard.json")
        csv_path = os.path.join(self.output_dir, "scorecard.csv")
        md_path = os.path.join(self.output_dir, "scorecard_report.md")

        MetricsCalculator.save_scorecard_json(metrics, json_path)
        MetricsCalculator.save_scorecard_csv(metrics, csv_path)

        run_info = {"scenario": eval_name, "duration": duration_total}
        md_content = MetricsCalculator.format_markdown_report(metrics, run_info)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"[*] Saved Scorecard JSON: {json_path}")
        print(f"[*] Saved Scorecard CSV:  {csv_path}")
        print(f"[*] Saved Scorecard MD:   {md_path}")

        # 2. Generate Matplotlib Error Plots
        plot_paths = PlotGenerator.generate_plots(combined_history, metrics, self.plots_dir)
        print(f"[*] Generated {len(plot_paths)} Diagnostic Plots in: {self.plots_dir}")

        # 3. Render Demonstration MP4 Video
        video_paths: List[str] = []
        if self.render_video_flag:
            xml_path = os.path.abspath("src/balancebot_description/worlds/skatepark.xml")
            if not os.path.exists(xml_path):
                # Search upwards
                cand_xml = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "balancebot_description",
                    "worlds",
                    "skatepark.xml",
                )
                if os.path.exists(cand_xml):
                    xml_path = cand_xml

            if os.path.exists(xml_path):
                video_file = os.path.join(self.output_dir, "balancebot_demo.mp4")
                print(f"[*] Rendering Headless MuJoCo MP4 Video ({fps} FPS)...")
                PlotGenerator.render_video(
                    history=combined_history,
                    xml_path=xml_path,
                    output_mp4=video_file,
                    fps=fps,
                    camera_name="tracking",
                )
                video_paths.append(video_file)
                print(f"[*] Rendered Demonstration Video: {video_file} ({os.path.getsize(video_file)} bytes)")

                # Also save standard name copy
                alt_video = os.path.join(self.output_dir, "balancebot_evaluation.mp4")
                shutil.copyfile(video_file, alt_video)
                video_paths.append(alt_video)
            else:
                print(f"[!] Warning: MuJoCo model xml not found at {xml_path}, skipping video rendering.")

        # 4. Mirror key artifacts to docs/ for documentation bundle
        docs_dir = os.path.abspath("docs")
        if os.path.exists(docs_dir) or os.path.exists(os.path.dirname(docs_dir)):
            os.makedirs(docs_dir, exist_ok=True)
            try:
                shutil.copyfile(json_path, os.path.join(docs_dir, "scorecard.json"))
                shutil.copyfile(csv_path, os.path.join(docs_dir, "scorecard.csv"))
                dash_src = os.path.join(self.plots_dir, "evaluation_dashboard.png")
                if os.path.exists(dash_src):
                    shutil.copyfile(dash_src, os.path.join(docs_dir, "evaluation_dashboard.png"))
                if video_paths:
                    shutil.copyfile(video_paths[0], os.path.join(docs_dir, "balancebot_demo.mp4"))
                print(f"[*] Mirrored scorecard, plots, and demo video to {docs_dir}/")
            except Exception as e:
                print(f"[!] Mirrored copy warning: {e}")

        # Console Summary
        print("----------------------------------------------------------------------")
        print(f"Overall Status: {'PASSED' if metrics['all_passed'] else 'FAILED'}")
        print(f"Steady-State Pitch:  {metrics['steady_pitch_deg']:.3f}° (< 5.0°) -> {'PASS' if metrics['steady_pitch_pass'] else 'FAIL'}")
        print(f"Velocity Error J_v:  {metrics['j_v']:.4f} m/s·s (<= 1.5)  -> {'PASS' if metrics['j_v_pass'] else 'FAIL'}")
        print(f"Pitch Error J_theta: {metrics['j_theta']:.4f} rad·s (<= 0.15) -> {'PASS' if metrics['j_theta_pass'] else 'FAIL'}")
        print(f"Settling Time T_s:   {metrics['settling_time']:.3f} s (< 2.0 s)   -> {'PASS' if metrics['settling_time_pass'] else 'FAIL'}")
        print(f"Chassis Clearance:   {metrics['min_clearance']:.4f} m (> 0.0 m)   -> {'PASS' if metrics['clearance_pass'] else 'FAIL'}")
        print(f"Landing Recovery:    {metrics['recovery_time']:.3f} s (<= 2.5 s) -> {'PASS' if metrics['recovery_pass'] else 'FAIL'}")
        print("----------------------------------------------------------------------")

        return metrics


def main(args: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="BalanceBot Headless Evaluation Runner")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "cruise", "flat", "ramp_jump", "jump"],
        help="Evaluation scenario (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_output",
        help="Target output directory for artifacts (default: eval_output)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video rendering",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video frame rate (default: 30)",
    )

    parsed = parser.parse_args(args)
    runner = EvaluateRunner(
        output_dir=parsed.output_dir,
        render_video=not parsed.no_video,
    )
    results = runner.run_evaluation(
        scenario=parsed.scenario,
        fps=parsed.fps,
    )

    sys.exit(0 if results.get("all_passed") else 1)


if __name__ == "__main__":
    main()
