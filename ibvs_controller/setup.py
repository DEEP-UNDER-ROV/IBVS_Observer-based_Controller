from setuptools import setup
 
package_name = 'ibvs_controller'
 
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='matthew',
    maintainer_email='matthew.troy271@email.com',
    description='IBVS controller for Underwater-ROV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ibvs_controll_node = ibvs_controller.ibvs_controll_node:main',
        ],
    },
)
