#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import OverrideRCIn


class HeaveTrimNode(Node):

    def __init__(self):
        super().__init__("heave_trim_node")

        self.pub = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
            10
        )

        # runtime adjustable parameter
        self.declare_parameter("bias", 0)

        self.timer = self.create_timer(
            1.0 / 20.0,
            self.send_rc
        )

        self.get_logger().info(
            "\n=== HEAVE TRIM MODE ===\n"
            "Vehicle outputs neutral PWM\n"
            "Only HEAVE channel affected\n\n"
            "Tune using:\n"
            "ros2 param set /heave_trim_node bias VALUE\n"
        )

    # -----------------------------------------
    def send_rc(self):

        bias = self.get_parameter(
            "bias"
        ).get_parameter_value().integer_value

        pwm = [1500] * 18

        # ArduSub mapping (your setup)
        pwm[2] = int(1500 + bias)

        msg = OverrideRCIn()
        msg.channels = pwm

        self.pub.publish(msg)

        self.get_logger().info(
            f"Heave PWM = {pwm[2]} (bias={bias})",
            throttle_duration_sec=0.5
        )


def main():
    rclpy.init()
    node = HeaveTrimNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import OverrideRCIn


class HeaveTrimNode(Node):

    def __init__(self):
        super().__init__("heave_trim_node")

        self.pub = self.create_publisher(
            OverrideRCIn,
            "/mavros/rc/override",
            10
        )

        # runtime adjustable parameter
        self.declare_parameter("bias", 0)

        self.timer = self.create_timer(
            1.0 / 20.0,
            self.send_rc
        )

        self.get_logger().info(
            "\n=== HEAVE TRIM MODE ===\n"
            "Vehicle outputs neutral PWM\n"
            "Only HEAVE channel affected\n\n"
            "Tune using:\n"
            "ros2 param set /heave_trim_node bias VALUE\n"
        )

    # -----------------------------------------
    def send_rc(self):

        bias = self.get_parameter(
            "bias"
        ).get_parameter_value().integer_value

        pwm = [1500] * 18

        # ArduSub mapping (your setup)
        pwm[2] = int(1500 + bias)

        msg = OverrideRCIn()
        msg.channels = pwm

        self.pub.publish(msg)

        self.get_logger().info(
            f"Heave PWM = {pwm[2]} (bias={bias})",
            throttle_duration_sec=0.5
        )


def main():
    rclpy.init()
    node = HeaveTrimNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
