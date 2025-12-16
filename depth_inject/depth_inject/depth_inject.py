#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class DepthPoseInject(Node):
    def __init__(self):
        super().__init__('depth_pose_inject')

        self.pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        self.timer = self.create_timer(0.1, self.publish)

    def publish(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # XY unused for now
        msg.pose.position.x = 0.0
        msg.pose.position.y = 0.0

        # DEPTH in meters (positive DOWN)
        msg.pose.position.z = 0.30

        # Identity orientation
        msg.pose.orientation.w = 1.0

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = DepthPoseInject()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
