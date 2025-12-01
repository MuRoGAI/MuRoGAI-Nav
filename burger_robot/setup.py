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

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='suraj',
    maintainer_email='surajb@iitgn.ac.in',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'burger_llm = burger_robot.burger_llm:main',
            'find_server = burger_robot.find_server:main',
            'goto_server = burger_robot.goto_server:main',
        ],
    },
)
