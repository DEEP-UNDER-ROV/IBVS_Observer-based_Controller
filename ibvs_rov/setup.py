import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ibvs'

setup(
    name=package_name,
    version='0.1',
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
            'pnp1 = ibvs.pnp1:main',
            'pnp2 = ibvs.pnp2:main',
            'ibvs1 = ibvs.ibvs1:main',
            'ibvs2 = ibvs.ibvs2:main',
            'ibvs3 = ibvs.ibvs3:main',
        ],
    },
)
