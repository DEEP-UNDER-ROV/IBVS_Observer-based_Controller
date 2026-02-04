#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State

# ================= CONFIG =================
# MODE = "velocity"     # "velocity" or "position"
MODE = "position"

# Velocity mode (m/s, rad/s)
VEL_SURGE = 0.0
VEL_SWAY  = 1.0       # <<< TEST VALUE
VEL_HEAVE = 0.0
VEL_YAW   = 0.0

# Position mode (meters, radians)
POS_X = 0.0
POS_Y = -5.0           # <<< SAME DIRECTION AS SWAY
POS_Z = 0.0
YAW   = 0.0

SETPOINT_RATE = 10.0  # Hz (MUST be > 2 Hz for PX4)
# =========================================


class EKFIsolationTest(Node):
    def __init__(self):
        super().__init__("ekf_isolation_test")

        self.state = State()

        self.state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self.state_cb,
            10
        )

        self.vel_pub = self.create_publisher(
            Twist,
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
            10
        )

        self.pos_pub = self.create_publisher(
            PoseStamped,
            "/mavros/setpoint_position/local",
            10
        )

        self.timer = self.create_timer(
            1.0 / SETPOINT_RATE,
            self.timer_cb
        )

        self.get_logger().info(f"EKF isolation node running in [{MODE.upper()}] mode")

    def state_cb(self, msg):
        self.state = msg

    def timer_cb(self):
        if not self.state.connected:
            return

        if MODE == "velocity":
            self.publish_velocity()
        elif MODE == "position":
            self.publish_position()

    # ------------------------------------------------
    def publish_velocity(self):
        msg = Twist()

        # Body-frame velocities (FLU)
        msg.linear.x = VEL_SURGE
        msg.linear.y = VEL_SWAY
        msg.linear.z = VEL_HEAVE

        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = VEL_YAW

        self.vel_pub.publish(msg)

    # ------------------------------------------------
    def publish_position(self):
        msg = PoseStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.pose.position.x = POS_X
        msg.pose.position.y = POS_Y
        msg.pose.position.z = POS_Z

        # Flat yaw only
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.pos_pub.publish(msg)


def main():
    rclpy.init()
    node = EKFIsolationTest()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
