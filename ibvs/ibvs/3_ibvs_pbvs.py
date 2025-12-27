#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PoseStamped, Vector3Stamped, Point
from std_msgs.msg import Float32MultiArray
from mavros_msgs.msg import PositionTarget
from scipy.spatial.transform import Rotation as R

from ibvs.constants import *

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")

        # --- Desired Position x,y,z (meters) ---
        self.tvec_des = np.array([0.0, 0.0, 1.2])  

        # --- Gains ---
        self.kp = np.diag([0.8, 0.8, 0.6])

        # --- State ---
        self.latest_pose = None
        self.latest_tvec = None

        # --- Subscriber ---
        self.sub = self.create_subscription(PoseStamped,"/mavros/vision_pose/pose",self.cb_pose,10)
        self.sub_tvec = self.create_subscription(Vector3Stamped, "/pnp/tvec", self.cb_tvec, 10)

        # --- Publisher ---
        self.pub = self.create_publisher(PositionTarget,"/mavros/setpoint_raw/local",10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)
        
        self.timer = self.create_timer(0.05, self.loop)
        self.get_logger().info("IBVS Scenario 3 (Position-based) started")

    def cb_pose(self, msg):
        self.latest_pose = msg

    def cb_tvec(self, msg):
        self.latest_tvec = np.array([msg.vector.x,
                                     msg.vector.y,
                                     msg.vector.z])

    def loop(self):
        if self.latest_pose is None or self.latest_tvec is None:
            return

        # --- EKF pose ---
        p_rov = self.latest_pose.pose.position
        q = self.latest_pose.pose.orientation

        # Body → Local ENU rotation
        R_BL = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        # tvec: tag position wrt camera (OpenCV frame)
        e_cam = self.latest_tvec - self.tvec_des   # [Xc, Yc, Zc]

        # Camera → Body (NED body)
        e_body = R_CB @ e_cam + P_CB

        # Body → Local ENU
        e_local = R_BL @ e_body

        # Desired absolute position (ENU)
        p_enu = np.array([p_rov.x, p_rov.y, p_rov.z])

        # --- Desired absolute position ---
        p_des_enu = p_enu - self.kp @ e_local

        # --- Publish setpoint ---
        cmd = PositionTarget()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.coordinate_frame = PositionTarget.FRAME_LOCAL_NED

        cmd.type_mask = (
            PositionTarget.IGNORE_VX |
            PositionTarget.IGNORE_VY |
            PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX |
            PositionTarget.IGNORE_AFY |
            PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )

        # ENU → NED
        cmd.position.x =  p_des_enu[0]   # North
        cmd.position.y = -p_des_enu[1]   # East
        cmd.position.z = -p_des_enu[2]   # Down

        cmd.yaw = 0.0

        self.pub.publish(cmd)

        # Debug
        err_msg = Float32MultiArray()
        err_msg.data = e_cam.tolist()
        self.err_pub.publish(err_msg)

def main():
    rclpy.init()
    node = IBVSControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
