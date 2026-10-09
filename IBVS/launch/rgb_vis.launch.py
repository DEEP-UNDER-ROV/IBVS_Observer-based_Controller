import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    mavros_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'mavros', 'apm.launch', 'fcu_url:=serial:///dev/ttyACM0:921600', 'gcs_url:=udp://@192.168.2.1:14550'],
        output='screen')

    realsense_rgb_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'realsense2_camera', 'rs_launch.py'],
        output='screen')

    visualisation_launch = ExecuteProcess(
        cmd=['ros2', 'run', 'ibvs', 'visualize'],
        output='screen')

    return LaunchDescription([
        mavros_launch,
        realsense_rgb_launch,
        visualisation_launch
    ])