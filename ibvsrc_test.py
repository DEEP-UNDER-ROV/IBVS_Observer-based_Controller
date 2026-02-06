#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.msg import OverrideRCIn

class RCTest(Node):
    def __init__(self):
        super().__init__("rc_test")

        self.pub = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
            10
        )

        self.create_timer(0.1, self.send_rc)  # 10 Hz

        self.get_logger().info("RC TEST NODE STARTED")

    def send_rc(self):
        rc = OverrideRCIn()
        rc.channels = [1500] * 18

        rc.channels[5] = 1700  # SWAY RIGHT (CHANGE THIS)

        self.get_logger().info(f"Publishing RC: CH5={rc.channels[5]}")
        self.pub.publish(rc)

def main():
    rclpy.init()
    rclpy.spin(RCTest())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
