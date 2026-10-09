import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    ibvs_launch = ExecuteProcess(
        cmd=['ros2', 'run', 'ibvs', 'ibvs2'],
        output='screen')

    visualisation_launch = ExecuteProcess(
        cmd=['ros2', 'run', 'ibvs', 'visualize'],
        output='screen')

    return LaunchDescription([
        ibvs_launch,
        visualisation_launch
    ])