from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
        (os.path.join('share', package_name, 'saved_paths'), glob('saved_paths/*')),
        (os.path.join('share', package_name, 'trajectory_logs'), glob('trajectory_logs/*')),
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
            'path_planner_node = path_planner.path_planner_node8:main',
            'path_planner_node1 = path_planner.path_planner_node11:main',

        ],
    },
)
