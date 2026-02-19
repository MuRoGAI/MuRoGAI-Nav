from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'navigation_manager'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), glob('data/*')),
        (os.path.join('share', package_name, 'data1'), glob('data1/*')),
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
            'navigation_manager = navigation_manager.navigation_manager:main',

            'navigation_manager_test1 = navigation_manager.navigation_manager_test1:main',
            'navigation_manager_test2 = navigation_manager.navigation_manager_test2:main',
            'test1 = navigation_manager.test1:main',
            'test2 = navigation_manager.test2:main',

        ],
    },
)
