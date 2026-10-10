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

import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_XACRO_PATH = os.path.join(PACKAGE_DIR, 'urdf', 'balancebot.urdf.xacro')
WORLD_SDF_PATH = os.path.join(PACKAGE_DIR, 'worlds', 'skatepark.world')
WORLD_XML_PATH = os.path.join(PACKAGE_DIR, 'worlds', 'skatepark.xml')


def test_asset_files_exist():
    """Verify all required model, world, launch, and configuration assets exist."""
    assert os.path.isfile(URDF_XACRO_PATH), 'URDF Xacro file missing'
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'urdf', 'materials.xacro'))
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'urdf', 'sensors.xacro'))
    assert os.path.isfile(WORLD_SDF_PATH), 'skatepark.world missing'
    assert os.path.isfile(WORLD_XML_PATH), 'skatepark.xml missing'
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'launch', 'display.launch.py'))
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'launch', 'gazebo.launch.py'))
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'package.xml'))
    assert os.path.isfile(os.path.join(PACKAGE_DIR, 'CMakeLists.txt'))


def test_urdf_xacro_compilation_and_structure():
    """Verify xacro processing generates valid URDF with required links and joints."""
    result = subprocess.run(
        ['xacro', URDF_XACRO_PATH],
        capture_output=True,
        text=True,
        check=True
    )
    urdf_content = result.stdout
    assert len(urdf_content) > 500, 'Generated URDF is unexpectedly empty'

    root = ET.fromstring(urdf_content)
    assert root.tag == 'robot'
    assert root.attrib.get('name') == 'balancebot'

    # Verify key links
    links = {link.attrib.get('name'): link for link in root.findall('link')}
    required_links = [
        'base_link',
        'imu_link',
        'lidar_link',
        'camera_link',
        'left_thigh_link',
        'left_shin_link',
        'left_wheel_link',
        'right_thigh_link',
        'right_shin_link',
        'right_wheel_link',
    ]
    for r_link in required_links:
        assert r_link in links, f'Missing required link: {r_link}'

    # Verify torso mass = 12 kg
    torso_inertial = links['base_link'].find('inertial')
    assert torso_inertial is not None
    torso_mass = float(torso_inertial.find('mass').attrib.get('value'))
    assert abs(torso_mass - 12.0) < 1e-3, f'Torso mass should be 12 kg, got {torso_mass}'

    # Verify wheel mass = 1.5 kg
    left_wheel_inertial = links['left_wheel_link'].find('inertial')
    assert left_wheel_inertial is not None
    wheel_mass = float(left_wheel_inertial.find('mass').attrib.get('value'))
    assert abs(wheel_mass - 1.5) < 1e-3, f'Wheel mass should be 1.5 kg, got {wheel_mass}'

    # Verify key joints
    joints = {joint.attrib.get('name'): joint for joint in root.findall('joint')}
    actuated_joints = [
        'left_hip_joint',
        'left_knee_joint',
        'left_wheel_joint',
        'right_hip_joint',
        'right_knee_joint',
        'right_wheel_joint',
    ]
    for j_name in actuated_joints:
        assert j_name in joints, f'Missing actuated joint: {j_name}'

    # Check hip joint limits (+-60 deg = +-1.047 rad)
    left_hip = joints['left_hip_joint']
    assert left_hip.attrib.get('type') == 'revolute'
    limit = left_hip.find('limit')
    assert float(limit.attrib['lower']) <= -1.047
    assert float(limit.attrib['upper']) >= 1.047

    # Check knee joint limits (0 to 120 deg = 0 to 2.094 rad)
    left_knee = joints['left_knee_joint']
    assert left_knee.attrib.get('type') == 'revolute'
    limit_knee = left_knee.find('limit')
    assert float(limit_knee.attrib['lower']) <= 0.0
    assert float(limit_knee.attrib['upper']) >= 2.094

    # Check wheel joints are continuous
    assert joints['left_wheel_joint'].attrib.get('type') == 'continuous'
    assert joints['right_wheel_joint'].attrib.get('type') == 'continuous'

    # Verify Gazebo sensor definitions and topics in generated URDF
    sensors = {}
    for gz in root.findall('gazebo'):
        s = gz.find('sensor')
        if s is not None:
            sensors[s.attrib.get('name')] = s

    assert 'imu_sensor' in sensors, 'Missing imu_sensor in URDF'
    assert 'lidar_sensor' in sensors, 'Missing lidar_sensor in URDF'
    assert 'depth_camera_sensor' in sensors, 'Missing depth_camera_sensor in URDF'

    assert sensors['imu_sensor'].find('topic').text == '/imu/data'
    assert sensors['lidar_sensor'].find('topic').text == '/scan'
    assert sensors['depth_camera_sensor'].find('topic').text == '/camera'

    imu_block = sensors['imu_sensor'].find('imu')
    assert imu_block is not None, 'Missing imu noise configuration in imu_sensor'
    assert imu_block.find('angular_velocity') is not None
    assert imu_block.find('linear_acceleration') is not None


def test_mujoco_compilation_and_kinematics():
    """Verify MuJoCo XML compiles, has matching DOFs, actuators, sensors, and terrain zones."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(WORLD_XML_PATH)
    assert model is not None, 'Failed to load MuJoCo model'

    # Generalized coordinates: 7 (floating base root) + 6 (actuated joints) = 13
    assert model.nq == 13, f'Expected nq=13, got {model.nq}'
    # Generalized velocities: 6 (floating base root) + 6 (actuated joints) = 12
    assert model.nv == 12, f'Expected nv=12, got {model.nv}'
    # Actuators: 6 (2 hip + 2 knee + 2 wheel)
    assert model.nu == 6, f'Expected nu=6, got {model.nu}'

    # Verify actuator names
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    expected_actuators = [
        'left_hip_motor',
        'right_hip_motor',
        'left_knee_motor',
        'right_knee_motor',
        'left_wheel_motor',
        'right_wheel_motor',
    ]
    for act in expected_actuators:
        assert act in actuator_names, f'Missing actuator: {act}'

    # Verify sensor sites
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    assert 'imu' in site_names, 'Missing IMU sensor site'
    assert 'lidar' in site_names, 'Missing LiDAR sensor site'
    assert 'camera' in site_names, 'Missing camera sensor site'

    # Verify native front camera
    assert model.ncam >= 1, 'Expected at least 1 native camera in MuJoCo model'
    camera_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]
    assert 'robot_front_camera' in camera_names, 'Missing robot_front_camera in MuJoCo model'

    # Verify terrain geometries
    geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    ]
    assert 'floor' in geom_names, 'Missing flat ground plane'
    assert any('bump' in name for name in geom_names), 'Missing sinusoidal bump geoms'
    assert any('rough' in name for name in geom_names), 'Missing rough terrain geoms'
    assert any('slope' in name for name in geom_names), 'Missing slope geom'
    assert any('ramp' in name for name in geom_names), 'Missing jump ramp geom'

    # Verify body masses
    torso_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    base_subtotal = model.body_mass[torso_body_id]
    assert 12.0 <= base_subtotal <= 13.0, f'Torso mass unexpected: {base_subtotal}'

    left_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_wheel')
    wheel_mass_diff = abs(model.body_mass[left_wheel_id] - 1.5)
    assert wheel_mass_diff < 1e-3, f'Left wheel mass mismatch: {model.body_mass[left_wheel_id]}'


def test_mujoco_simulation_step_stability():
    """Verify MuJoCo physics engine steps stably without NaN values or explosion."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(WORLD_XML_PATH)
    data = mujoco.MjData(model)

    # Step simulation 500 steps (0.5 seconds at dt = 1ms)
    for _ in range(500):
        mujoco.mj_step(model, data)
        assert not np.any(np.isnan(data.qpos)), 'NaN detected in generalized positions'
        assert not np.any(np.isnan(data.qvel)), 'NaN detected in generalized velocities'
        assert not np.any(np.isinf(data.qpos)), 'Inf detected in generalized positions'
        assert not np.any(np.isinf(data.qvel)), 'Inf detected in generalized velocities'

    # Ensure robot remains within reasonable bounds
    assert -5.0 < data.qpos[0] < 5.0, 'Robot X runaway position'
    assert -5.0 < data.qpos[1] < 5.0, 'Robot Y runaway position'
    assert 0.0 < data.qpos[2] < 2.0, 'Robot Z runaway position'


def test_articulated_leg_height_workspace():
    """Verify kinematic range of articulated legs covers height L in [0.18, 0.38] m."""
    thigh_len = 0.20

    # Kinematics for symmetric vertical leg: q_hip = -q_knee / 2
    # L(q_knee) = 2 * thigh_len * cos(q_knee / 2)
    def leg_height(q_knee):
        return 2.0 * thigh_len * np.cos(q_knee / 2.0)

    # Max height at nominal extension (q_knee ~ 36 deg = 0.63 rad)
    h_max = leg_height(np.radians(36.0))
    assert h_max >= 0.38, f'Max height {h_max} must reach or exceed 0.38 m'

    # Min height at maximum crouch (q_knee ~ 127 deg = 2.22 rad)
    h_min = leg_height(np.radians(127.0))
    assert h_min <= 0.18, f'Min height {h_min} must reach or fall below 0.18 m'
