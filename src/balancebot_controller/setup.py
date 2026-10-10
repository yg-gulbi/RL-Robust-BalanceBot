from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'balancebot_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RL-Robust-BalanceBot Team',
    maintainer_email='dev@balancebot.org',
    description='Gain-Scheduled LQR-I balance controller, estimator and teleop for BalanceBot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'balance_controller = balancebot_controller.balance_controller:main',
            'leg_kinematics_controller = balancebot_controller.leg_kinematics:main',
            'state_estimator = balancebot_controller.state_estimator:main',
            'terrain_observer = balancebot_controller.terrain_observer:main',
            'terrain_observer_node = balancebot_controller.terrain_observer:main',
            'teleop_node = balancebot_controller.teleop_node:main',
            'gazebo_actuator_bridge = balancebot_controller.gazebo_actuator_bridge:main',
        ],
    },
)
