import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    home_dir = os.path.expanduser('~')
    
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    bag_output_path = os.path.join(home_dir, f'rosbag_{timestamp}')

    record_process = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '-o', bag_output_path,
            '--max-cache-size', '100000000',
            '/apriltag/corners',
            '/detection1',
            '/detection2',
            '/ibvs/error/px',
            '/ibvs/error/no',
            '/ibvs/ukf/data',
            '/ibvs/nu_B_hat',
            '/ibvs/pos_hat',
            '/ibvs/torque',
            '/mavros/imu/data',
            '/mavros/imu/data_raw',
            '/camera/camera/accel/sample',
            '/camera/camera/gyro/sample',
            '/camera/camera/imu',
            '/mavros/rc/override',
            '/camera/overlay/image_raw/compressed',
        ], output='screen'
    )

    return LaunchDescription([
        record_process
    ])
