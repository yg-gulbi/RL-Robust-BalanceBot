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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('balancebot_description')
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')
    default_world_path = PathJoinSubstitution([pkg_share, 'worlds', 'skatepark.world'])
    xacro_file = PathJoinSubstitution([pkg_share, 'urdf', 'balancebot.urdf.xacro'])

    robot_description_content = Command(
        [
            FindExecutable(name='xacro'), ' "',
            xacro_file, '"'
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    # Gazebo Sim launch
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': ['-r "', LaunchConfiguration('world'), '"']}.items(),
    )

    # Robot spawner in Gazebo
    spawner = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_balancebot',
        output='screen',
        arguments=[
            '-name', 'balancebot',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('spawn_x'),
            '-y', LaunchConfiguration('spawn_y'),
            '-z', LaunchConfiguration('spawn_z'),
        ],
    )

    pkg_controller = FindPackageShare('balancebot_controller')
    controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_controller, 'launch', 'controllers.launch.py'])
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'world',
            default_value=default_world_path,
            description='World file path'
        ),
        DeclareLaunchArgument(
            'spawn_x',
            default_value='0.0',
            description='Initial robot X position'
        ),
        DeclareLaunchArgument(
            'spawn_y',
            default_value='0.0',
            description='Initial robot Y position'
        ),
        DeclareLaunchArgument(
            'spawn_z',
            default_value='0.0',
            description='Initial robot Z position'
        ),
        # Start Gazebo Sim
        gz_sim,
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description, {'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
        # Spawn robot into Gazebo
        spawner,
        # ROS-Gazebo Bridge (Bridges clock, IMU, LiDAR, and Camera)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            output='screen',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/model/balancebot/joint/left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                '/model/balancebot/joint/right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                '/model/balancebot/joint/left_hip_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                '/model/balancebot/joint/left_knee_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                '/model/balancebot/joint/right_hip_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
                '/model/balancebot/joint/right_knee_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            ],
            remappings=[
                ('/camera/image', '/camera/image_raw'),
                ('/camera/depth_image', '/camera/depth/image_raw'),
            ],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        # Launch BalanceBot Controllers (LQR-I, VMC, FSM)
        controllers_launch,
        # Launch Gazebo Actuator Bridge (Torque MultiArray -> Joint Forces)
        Node(
            package='balancebot_controller',
            executable='gazebo_actuator_bridge',
            name='gazebo_actuator_bridge',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
