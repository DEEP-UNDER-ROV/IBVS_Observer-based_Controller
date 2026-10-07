import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    realsense_rgb_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'realsense2_camera', 'rs_launch.py'],
        output='screen')

    visualisation_launch = ExecuteProcess(
        cmd=['ros2', 'run', 'ibvs', 'visualize'],
        output='screen')

    return LaunchDescription([
        realsense_rgb_launch,
        visualisation_launch
    ])