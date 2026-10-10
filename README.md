# RL-Robust-BalanceBot

> **Two-Wheeled Active-Leg Self-Balancing Robot in ROS 2 & Gazebo/MuJoCo Simulation**  
> *Extreme Terrain Traversal, Dynamic Equilibrium LQR Control, Jump Landing Shock Absorption, and Real-Time 3D Teleoperation*

[![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy%20Jalisco-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Physics](https://img.shields.io/badge/Physics-Gazebo%20Harmonic%20%7C%20MuJoCo%203.14-orange.svg)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-407%20Passed-brightgreen.svg)]()

---

## 📌 Project Overview

**RL-Robust-BalanceBot** is an advanced robotic simulation system featuring a 6-DOF two-wheeled active-leg self-balancing robot designed to navigate extreme skatepark environments. It combines high-speed multi-body dynamics, analytic continuous-time **Gain-Scheduled LQR-I balance control**, **Virtual Model Control (VMC)** for articulated legs, and a **5-state flight finite-state machine (FSM)** for ramp jump and landing shock absorption.

![Evaluation Dashboard](docs/evaluation_dashboard.png)

### Key Features
- **6-DOF Hybrid Kinematic/Dynamic Model**: 2 continuous drive wheels + 4 revolute pitch leg joints (Hip/Knee).
- **Gain-Scheduled LQR-I Inverted Pendulum Control**: Solves Algebraic Riccati Equations (ARE) parameterised by effective leg length $L \in [0.18, 0.38]\,\text{m}$ at 100 Hz.
- **Active 2-DOF Sagittal Leg Kinematics (VMC)**: Independent height and fore/aft wheel offset adjustment with dynamic equilibrium pitch trimming ($\theta_{eq} = \arcsin(\Delta x / L_{eff})$).
- **5-State Flight & Shock Dissipation FSM**: Free-fall detection ($||\mathbf{a}|| < 2.5\,\text{m/s}^2$), torque suppression in mid-air, and impact shock absorption ($||\mathbf{a}|| > 15\,\text{m/s}^2$).
- **Multi-Sensor Simulation**: 200 Hz 6-axis IMU, 20 Hz 360° LiDAR, and 30 Hz RGB-D Depth Camera.
- **Dual Physics Engine Support**:
  - **ROS 2 Jazzy + Gazebo Sim (Harmonic)** via `ros_gz_bridge`.
  - **Real-Time Interactive MuJoCo (3.14)** 3D Viewer with smooth teleoperation and floating controller GUI.

---

## 🏗 System Architecture

```
       [ LiDAR (20 Hz) ]      [ RGB-D Depth Camera (30 Hz) ]
                 \                  /
            +----------------------------+
            |      Torso Base (12 kg)    |
            |     IMU Sensor (200 Hz)    |
            +--------------+-------------+
                  /                 \
        [ Left Hip Joint ]    [ Right Hip Joint ]  (Revolute Pitch, +/-1.2 rad, tau_max = 50 N*m)
                 |                   |
          Thigh Link (0.2m)    Thigh Link (0.2m)
                 |                   |
        [ Left Knee Joint ]   [ Right Knee Joint ] (Revolute Pitch, 0 to 2.3 rad, tau_max = 50 N*m)
                 |                   |
          Shank Link (0.2m)   Shank Link (0.2m)
                 |                   |
        [ Left Wheel Hub ]    [ Right Wheel Hub ] (Continuous Pitch, tau_max = 25 N*m)
            (Radius = 0.1m, Track Width = 0.45m)
```

For complete mathematical derivations and interface specifications, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🚀 Quick Start & Installation

### Prerequisites
- **Ubuntu 24.04 LTS (Noble)**
- **ROS 2 Jazzy Jalisco**
- **Python 3.12+**
- **MuJoCo 3.14+** (`pip install mujoco glfw`)

```bash
# Clone the repository
git clone https://github.com/yg-gulbi/RL-Robust-BalanceBot.git
cd RL-Robust-BalanceBot

# Install dependencies & build ROS 2 workspace
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 🎮 Running Interactive Simulation

### Option 1: Interactive MuJoCo 3D Simulator (Recommended)

Runs high-speed real-time 3D physics with interactive keyboard driving and a floating remote controller GUI:

```bash
python3 run_mujoco_sim.py
```

#### Controls
| Key | Action | Details |
| :---: | :--- | :--- |
| **`↑` / `W` / `I`** | **Drive Forward** | Smoothly accelerates forward; holds speed continuously while driving |
| **`↓` / `S` / `,`** | **Drive Backward** | Smoothly accelerates backward |
| **`←` / `A` / `J`** | **Steer Left** | Natural differential turning |
| **`→` / `D` / `L`** | **Steer Right** | Natural differential turning |
| **`Space` / `K`** | **Emergency Brake** | Immediate brake and position locking (Station-Keeping) |
| **`P`** | **40 N Push Impulse** | Tests external disturbance rejection (Watch the robot recover balance!) |
| **`R`** | **Reset** | Resets robot pose to nominal standing stance |
| **`M`** | **Toggle Mode** | Switches between Auto-Stop on Release and Cruise Control |

---

### Option 2: ROS 2 + Gazebo Sim (Harmonic)

Runs the full ROS 2 ecosystem connected via `ros_gz_bridge`:

```bash
# Terminal 1: Launch Gazebo Sim with skatepark world & controllers
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch balancebot_description gazebo.launch.py

# Terminal 2: Run ROS 2 keyboard teleoperation node
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run balancebot_controller teleop_node
```

---

## 📊 Verification & Benchmarks

The controller has undergone automated verification across 407 authentic test cases (including boundary corners and adversarial external disturbances):

| Metric | Target Requirement | Measured Result | Status |
| :--- | :---: | :---: | :---: |
| **Steady-State Pitch Error** ($|\theta|$) | $< 5.0^\circ$ | **$1.84^\circ$** | ✅ PASSED |
| **Velocity Tracking Error** ($J_v$) | $< 0.15\,\text{m/s}$ | **$0.062\,\text{m/s}$** | ✅ PASSED |
| **40 N Push Impulse Settling Time** | $< 1.5\,\text{s}$ | **$0.82\,\text{s}$** | ✅ PASSED |
| **Chassis Ground Clearance** | $> 0.05\,\text{m}$ | **$0.091\,\text{m}$** | ✅ PASSED |
| **Ramp Jump Landing Recovery** | No Toppling ($|\theta| < 25^\circ$) | **$4.63^\circ$ max tilt** | ✅ PASSED |

Run the verification test suite:
```bash
pytest src/balancebot_evaluator/test/
```

---

## 📂 Repository Structure

```
.
├── run_mujoco_sim.py             # Interactive 3D MuJoCo simulator & teleop runner
├── ARCHITECTURE.md               # Detailed system architecture & math formulation
├── docs/                         # Demonstration MP4 video, scorecards & plots
│   ├── balancebot_demo.mp4
│   ├── evaluation_dashboard.png
│   └── scorecard.csv
└── src/
    ├── balancebot_description/    # URDF/Xacro, MJCF skatepark worlds & sensor plugins
    ├── balancebot_controller/     # LQR balance node, VMC leg kinematics, FSM observer
    └── balancebot_evaluator/      # Automated evaluation harness & test suites (407 tests)
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
