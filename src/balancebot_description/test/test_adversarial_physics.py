# Copyright 2026 RL-Robust-BalanceBot Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adversarial stress test suite for BalanceBot MuJoCo kinematics and physics."""

import os

import mujoco
import numpy as np
import pytest

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_XML_PATH = os.path.join(PACKAGE_DIR, 'worlds', 'skatepark.xml')


@pytest.fixture
def mujoco_env():
    """Load model with energy tracking enabled and return model, data."""
    model = mujoco.MjModel.from_xml_path(WORLD_XML_PATH)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    data = mujoco.MjData(model)
    return model, data


def _assert_numerically_sound(data, step_idx=0, context=''):
    """Assert qpos, qvel, and qacc contain no NaNs or Infs."""
    assert not np.any(np.isnan(data.qpos)), f'NaN in qpos at step {step_idx} ({context})'
    assert not np.any(np.isnan(data.qvel)), f'NaN in qvel at step {step_idx} ({context})'
    assert not np.any(np.isnan(data.qacc)), f'NaN in qacc at step {step_idx} ({context})'
    assert not np.any(np.isinf(data.qpos)), f'Inf in qpos at step {step_idx} ({context})'
    assert not np.any(np.isinf(data.qvel)), f'Inf in qvel at step {step_idx} ({context})'


def test_drop_from_height_1m_and_energy_boundedness(mujoco_env):
    """Adversarial test: Drop robot from 1.0m height, verify stability and bounded energy."""
    model, data = mujoco_env
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0  # 1.0m height

    max_ke = 0.0
    for step in range(3000):  # 3.0 seconds
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, '1m drop')
        ke = float(data.energy[1])
        if ke > max_ke:
            max_ke = ke

    # Robot must have hit ground and settled with bounded energy
    assert max_ke < 200.0, f'Kinetic energy exceeded safe threshold: {max_ke} J'
    assert data.energy[1] < 0.1, f'Robot failed to dissipate impact energy: {data.energy[1]} J'
    assert 0.0 < data.qpos[2] < 0.7, f'Final resting height unreasonable: {data.qpos[2]}'


@pytest.mark.parametrize('angle_deg,axis', [
    (45, [0, 1, 0]),     # Pitch 45 deg
    (90, [0, 1, 0]),     # Pitch 90 deg (torso flat)
    (180, [0, 1, 0]),    # Inverted upside-down
    (45, [1, 0, 0]),     # Roll 45 deg
    (90, [1, 0, 0]),     # Roll 90 deg (sideways drop)
    (60, [1, 1, 1]),     # Compound roll-pitch-yaw
])
def test_drop_multi_orientation_extremes(mujoco_env, angle_deg, axis):
    """Adversarial test: Drop from 1.0m with extreme pitch/roll/yaw orientations."""
    model, data = mujoco_env
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0

    # Quaternion from axis-angle
    ax = np.array(axis, dtype=float) / np.linalg.norm(axis)
    half = np.radians(angle_deg) / 2.0
    data.qpos[3:7] = [
        np.cos(half),
        ax[0] * np.sin(half),
        ax[1] * np.sin(half),
        ax[2] * np.sin(half)
    ]

    for step in range(2500):
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, f'Drop orientation {angle_deg}deg {axis}')

    # Energy must remain bounded
    assert data.energy[1] < 10.0, f'Excessive residual kinetic energy: {data.energy[1]} J'


def test_joint_angle_limits_and_over_limit_recovery(mujoco_env):
    """Adversarial test: Force joint angles to boundaries and beyond limits."""
    model, data = mujoco_env

    extreme_configs = [
        (1.20, 0.0, 1.20, 0.0),      # Upper hip, lower knee
        (-1.20, 2.30, -1.20, 2.30),  # Lower hip, upper knee
        (1.20, 2.30, -1.20, 0.0),    # Asymmetric scissors
        (1.30, -0.05, -1.30, 2.40),  # Over-limit initialization
    ]

    for l_hip, l_knee, r_hip, r_knee in extreme_configs:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.55
        data.qpos[7] = l_hip
        data.qpos[8] = l_knee
        data.qpos[10] = r_hip
        data.qpos[11] = r_knee

        for step in range(1500):
            mujoco.mj_step(model, data)
            _assert_numerically_sound(data, step, f'Joint cfg {l_hip},{l_knee}')

        # Joints should remain within valid mechanical bounds after solver steps
        assert -1.25 <= data.qpos[7] <= 1.25, f'Left hip limit violated: {data.qpos[7]}'
        assert -0.05 <= data.qpos[8] <= 2.35, f'Left knee limit violated: {data.qpos[8]}'


@pytest.mark.parametrize('wrench_desc,wrench_vector,duration_ms', [
    ('Horizontal push 100N', [100.0, 0, 0, 0, 0, 0], 200),
    ('Horizontal push -150N', [-150.0, 0, 0, 0, 0, 0], 200),
    ('Lateral impact 200N', [0, 200.0, 0, 0, 0, 0], 100),
    ('Vertical launch 300N', [0, 0, 300.0, 0, 0, 0], 100),
    ('Extreme strike 500N', [500.0, 0, 0, 0, 0, 0], 50),
    ('Extreme slam -500N', [0, 0, -500.0, 0, 0, 0], 50),
    ('Pitch torque 100Nm', [0, 0, 0, 0, 100.0, 0], 100),
    ('Roll torque 100Nm', [0, 0, 0, 100.0, 0, 0], 100),
    ('Yaw torque 100Nm', [0, 0, 0, 0, 0, 100.0], 100),
])
def test_extreme_external_force_impulses(mujoco_env, wrench_desc, wrench_vector, duration_ms):
    """Adversarial test: Apply external force impulses >50 N (up to 500 N, 100 Nm)."""
    model, data = mujoco_env
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.55
    for _ in range(100):
        mujoco.mj_step(model, data)

    steps_pulse = int(duration_ms)  # 1 ms per step
    for step in range(steps_pulse):
        data.xfrc_applied[base_id] = wrench_vector
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, f'Applying {wrench_desc}')

    data.xfrc_applied[base_id] = 0.0
    for step in range(1500):
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, f'Recovery after {wrench_desc}')

    assert data.energy[1] < 15.0, f'Residual energy high after {wrench_desc}: {data.energy[1]}'


def test_actuator_saturation_and_high_frequency_chatter(mujoco_env):
    """Adversarial test: Continuous actuator saturation (+-50 Nm) and 200 Hz bang-bang chatter."""
    model, data = mujoco_env

    # 1. Continuous positive saturation
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.55
    data.ctrl[:] = [50.0, 50.0, 50.0, 50.0, 25.0, 25.0]
    for step in range(1000):
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, 'Positive saturation')

    # 2. Continuous negative saturation
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.55
    data.ctrl[:] = [-50.0, -50.0, -50.0, -50.0, -25.0, -25.0]
    for step in range(1000):
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, 'Negative saturation')

    # 3. Bang-bang chattering (200 Hz toggle)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.55
    for step in range(1000):
        sign = 1.0 if (step // 5) % 2 == 0 else -1.0
        data.ctrl[:] = sign * np.array([50.0, 50.0, 50.0, 50.0, 25.0, 25.0])
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, 'Bang-bang chattering')


@pytest.mark.parametrize('obstacle_name,x,y,z', [
    ('Bump 1', 3.3, 0.0, 1.0),
    ('Rough block 2', 7.9, -0.3, 1.0),
    ('Slope incline', 13.0, -3.0, 1.5),
    ('Ramp lip edge', 14.0, 0.0, 1.5),
    ('Ramp board slope', 13.52, 0.0, 1.5),
])
def test_terrain_obstacle_drop_impact(mujoco_env, obstacle_name, x, y, z):
    """Adversarial test: Drop directly onto skatepark obstacles and verify contact stability."""
    model, data = mujoco_env
    mujoco.mj_resetData(model, data)
    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = z

    for step in range(2500):
        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, f'Obstacle drop on {obstacle_name}')

    assert data.energy[1] < 10.0, f'Energy failed to dissipate on {obstacle_name}'


def test_randomized_multi_perturbation_long_run(mujoco_env):
    """Adversarial test: 5000 simulation steps (5s) under random wrench pulses and motor inputs."""
    model, data = mujoco_env
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')

    np.random.seed(12345)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0
    data.qvel[3:6] = np.random.uniform(-4.0, 4.0, size=3)

    for step in range(5000):
        if step % 500 < 50:
            force = np.random.uniform(-200.0, 200.0, size=3)
            torque = np.random.uniform(-30.0, 30.0, size=3)
            data.xfrc_applied[base_id] = np.concatenate([force, torque])
        else:
            data.xfrc_applied[base_id] = 0.0

        if step % 100 == 0:
            data.ctrl[:] = np.random.uniform(-30.0, 30.0, size=6)
            data.ctrl[4:] = np.clip(data.ctrl[4:], -15.0, 15.0)

        mujoco.mj_step(model, data)
        _assert_numerically_sound(data, step, 'Multi-perturbation long run')
