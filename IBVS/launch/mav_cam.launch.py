import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    vision_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'rov_vision', 'uw_rs_apriltag_triangulation.launch.xml'],
        output='screen')

    disable_emitter = TimerAction(
        period=3.0, actions=[ExecuteProcess(
            cmd=['ros2', 'param', 'set', '/camera/camera', 'depth_module.emitter_enabled', '0'],
            output='screen')])

    mavros_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'mavros', 'apm.launch', 'fcu_url:=serial:///dev/ttyACM0:921600', 'gcs_url:=udp://@192.168.2.1:14550'],
        output='screen')

    return LaunchDescription([
        vision_launch,
        disable_emitter,
        mavros_launch
    ])