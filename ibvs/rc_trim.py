
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mavros_msgs.msg import OverrideRCIn


class RCTrimNode(Node):
    def __init__(self):
        super().__init__("rc_trim")
        self.pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)

        # Runtime-adjustable parameters
        self.declare_parameter("surge", 0)
        self.declare_parameter("sway",  0)
        self.declare_parameter("heave", 0)

        self.declare_parameter("roll",  0)
        self.declare_parameter("pitch", 0)
        self.declare_parameter("yaw",   0)

        self.channel_map = {
            "surge": 4,  # RC channel 5
            "sway":  5,  # RC channel 6
            "heave": 2,  # RC channel 3
            "roll":  1,  # RC channel 2
            "pitch": 0,  # RC channel 1
            "yaw":   3   # RC channel 4
        }

        # PWM configuration
        self.neutral_pwm = 1500
        self.min_pwm = 1100
        self.max_pwm = 1900

        self.timer = self.create_timer(1.0 / 20.0, self.send_rc)

        self.get_logger().info(
            "\n=== SIX-DOF RC TRIM MODE ===\n"
            "All channels initialized to neutral PWM.\n"
            "Runtime adjustable parameters:\n"
            "  surge, sway, heave\n"
            "  roll, pitch, yaw\n\n"
            "Example:\n"
            "ros2 param set /rc_trim channel 50\n"
        )

    def send_rc(self):
        pwm = [OverrideRCIn.CHAN_NOCHANGE] * 18

        for axis, channel in self.channel_map.items():
            bias = self.get_parameter(axis).value
            pwm_value = self.neutral_pwm + int(bias)
            pwm_value = max(self.min_pwm, 
                        min(self.max_pwm, pwm_value))
            pwm[channel] = pwm_value

        msg = OverrideRCIn()
        msg.channels = pwm
        self.pub.publish(msg)

        # Logging
        self.get_logger().info(
            "RC PWM | + | ".join(f"{axis}={pwm[ch]}" for axis, ch in self.channel_map.items()),
            throttle_duration_sec=0.5)


def main():
    rclpy.init()
    node = RCTrimNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
