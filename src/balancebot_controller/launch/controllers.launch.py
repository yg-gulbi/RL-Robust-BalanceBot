from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock if true'
        ),
        DeclareLaunchArgument(
            'tau_max',
            default_value='25.0',
            description='Maximum wheel torque limit (N*m)'
        ),
        DeclareLaunchArgument(
            'control_rate',
            default_value='100.0',
            description='Balance control loop frequency (Hz)'
        ),
        Node(
            package='balancebot_controller',
            executable='state_estimator',
            name='state_estimator_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        Node(
            package='balancebot_controller',
            executable='leg_kinematics_controller',
            name='leg_kinematics_controller_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        Node(
            package='balancebot_controller',
            executable='terrain_observer_node',
            name='terrain_observer_node',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
        Node(
            package='balancebot_controller',
            executable='balance_controller',
            name='balance_controller_node',
            output='screen',
            parameters=[
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'tau_max': LaunchConfiguration('tau_max'),
                    'control_rate': LaunchConfiguration('control_rate'),
                }
            ],
        ),
    ])
