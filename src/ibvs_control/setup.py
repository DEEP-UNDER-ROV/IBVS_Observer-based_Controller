import os
from glob import glob
from setuptools import setup

package_name = 'ibvs_control'

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
            'detector = ibvs_control.detector:main',
            'pnp = ibvs_control.pnp:main',
            'ibvs = ibvs_control.ibvs:main',
            'stream = ibvs_control.stream:main',
	    'pnp_new = ibvs_control.pnp_new:main',
        ],
    },
)
