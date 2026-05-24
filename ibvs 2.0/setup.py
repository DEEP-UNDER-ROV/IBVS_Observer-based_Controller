import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ibvs'

setup(
    name=package_name,
    version='0.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SCHW4RZELISTE',
    maintainer_email='matthew.troy271@gmail.com',
    description='IBVS Control for U-ROVs',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = ibvs.detector:main',
            'visualise = ibvs.visualisation:main',
            'ibvs = ibvs.ibvs:main',
            'heave_trim = ibvs.heave_trim:main',
        ],
    },
)
