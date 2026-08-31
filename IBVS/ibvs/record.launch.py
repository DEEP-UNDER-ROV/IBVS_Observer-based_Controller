import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Set output folder path
    data_dir = os.path.expanduser('$HOME')
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    bag_output_path = os.path.join(data_dir, f'rosbag_{timestamp}')
    qos_config_path = os.path.join(data_dir, '~/rov_ws/src/IBVS/qos_overrides.yaml')

    # Define the record command execution process
    record_process = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '-o', bag_output_path,
            '--max-cache-size', '100000000',
            '--qos-profile-overrides-path', qos_config_path,
            '/mavros/imu/data',
            '/mavros/rc/override',
            '/ibvs/ukf/data',
            '/ibvs/nu_B_hat',
            '/ibvs/torque',
            '/ibvs/error/px',
            '/ibvs/error/no',
            '/apriltag/corners',
            '/detection1',
            '/detection2',
            '/corrected/left/image_raw',
            '/camera/overlay/image_raw/compressed'
        ],
        output='screen'
    )

    return LaunchDescription([
        record_process
    ])
