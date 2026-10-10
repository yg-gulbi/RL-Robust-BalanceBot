from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'balancebot_evaluator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RL-Robust-BalanceBot Team',
    maintainer_email='dev@balancebot.org',
    description='Automated evaluation, quantitative metrics scorecard, and visualization runner for BalanceBot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'evaluate_runner = balancebot_evaluator.evaluate_runner:main',
        ],
    },
)
