#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget

from ibvs.constants import *

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")

        self.z_des = 1.2   # meters from tag
        self.x_des = 0.0
        self.y_des = 0.0

        # Gains
        self.kp = np.diag([0.8, 0.8, 0.6])

        self.sub = self.create_subscription(PoseStamped,"/mavros/vision_pose/pose",self.pose_cb,10)
        self.sub_tvec = self.create_subscription(Vector3Stamped, "/pnp/tvec", self.cb_tvec, 10)
        
        self.pub = self.create_publisher(PositionTarget,"/mavros/setpoint_raw/local",10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)
        
        self.timer = self.create_timer(0.05, self.loop)
        self.latest_pose = None
        self.get_logger().info("IBVS Scenario 3 (Position-based) started")

    def pose_cb(self, msg):
        self.latest_pose = msg

    def loop(self):
        if self.latest_pose is None:
            return

        p = self.latest_pose.pose.position

        # Relative ENU from PnP
        x_enu = p.x
        y_enu = p.y
        z_enu = p.z

        # Desired relative pose
        err_enu = np.array([
            x_enu - self.x_des,
            y_enu - self.y_des,
            z_enu - self.z_des
        ])

        # Convert ENU → NED
        err_ned = np.array([
            err_enu[0],
            -err_enu[1],
            -err_enu[2]
        ])

        cmd = PositionTarget()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        # Mask everything except POSITION
        cmd.type_mask = (
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        cmd.position.x = -self.kp[0,0] * err_ned[0]
        cmd.position.y = -self.kp[1,1] * err_ned[1]
        cmd.position.z = -self.kp[2,2] * err_ned[2]

        cmd.yaw = 0.0  # keep yaw stabilized by ArduSub

        self.pub.publish(cmd)

def main():
    rclpy.init()
    node = IBVSPositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
