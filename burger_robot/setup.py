from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'burger_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'trajectory_logs'), glob('trajectory_logs/*')),
        (os.path.join('share', package_name, 'trajectory_logs1'), glob('trajectory_logs1/*')),
        (os.path.join('share', package_name, 'trajectory_logs2'), glob('trajectory_logs2/*')),
        (os.path.join('share', package_name, 'trajectory_logs3'), glob('trajectory_logs3/*')),
        (os.path.join('share', package_name, 'trajectory_logs4'), glob('trajectory_logs4/*')),
        (os.path.join('share', package_name, 'trajectory_logs_1'), glob('trajectory_logs_1/*')),
        (os.path.join('share', package_name, 'trajectory_logs_2'), glob('trajectory_logs_2/*')),
        (os.path.join('share', package_name, 'trajectory_logs_3'), glob('trajectory_logs_3/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='multi-robot',
    maintainer_email='bhavishraib@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'burger_robot_llm = burger_robot.burger_robot_llm:main',

            'diff_drive_controller = burger_robot.controller:main',
            'diff_drive_controller1 = burger_robot.controller1:main',
            
            'fleet_controller_node = burger_robot.fleet_controller_node:main',

            'controller5_holo_pose = burger_robot.controller5_holo_pose:main',
            'controller5_holo_pose1 = burger_robot.controller5_holo_pose1:main',
            'controller5_holo_pose2 = burger_robot.controller5_holo_pose2:main',
            'controller7_diff_drive_pose = burger_robot.controller7_diff_drive_pose:main',
            'controller7_diff_drive_pose1 = burger_robot.controller7_diff_drive_pose1:main',
            'controller7_diff_drive_pose2 = burger_robot.controller7_diff_drive_pose2:main',

            'controller9 = burger_robot.controller9:main',
            'controller9_pub = burger_robot.controller9_pub:main',
            'controller9_pub1 = burger_robot.controller9_pub1:main',

            'robot_test = burger_robot.robot_test:main',
        ],
    },
)
