#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range

class DepthInjector(Node):
    def __init__(self):
        super().__init__('rangefinder_inject')

        self.pub = self.create_publisher(
            Range,
            '/mavros/distance_sensor/range',
            10
        )

        self.timer = self.create_timer(0.1, self.publish_range)

    def publish_range(self):
        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.radiation_type = Range.ULTRASOUND
        msg.field_of_view = 0.1
        msg.min_range = 0.2
        msg.max_range = 10.0

        msg.range = 0.30   # meters

        self.pub.publish(msg)

def main():
    rclpy.init()
    node = DepthInjector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
