from setuptools import find_packages, setup
from glob   import glob
import os

package_name = 'robot_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='name',
    maintainer_email='name@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_llm_node = robot_llm.robot_llm:main',
            'team_llm_node = robot_llm.team_llm:main',
            'team_llm_node1 = robot_llm.team_llm1:main',
            'team_llm_node2 = robot_llm.team_llm2:main',
            'team_llm_node3 = robot_llm.team_llm3:main',

            # 'mobile_robot = robot_llm.mobile_robot:main',
            # 'manipulator_robot = robot_llm.manipulator_robot:main',

            'multi_robot_odom_offset = robot_llm.multi_robot_odom_offset:main',
        ],
    },
)
