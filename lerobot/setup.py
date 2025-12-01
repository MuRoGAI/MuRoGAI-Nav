from setuptools import find_packages, setup
from glob import glob
import os
package_name = 'lerobot'

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
            'lerobot_llm = lerobot.lerobot_llm:main',
            'pick_server = lerobot.pick_server:main',
        ],
    },
)
