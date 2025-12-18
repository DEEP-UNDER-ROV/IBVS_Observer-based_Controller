#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs_control.constants import FX, FY, CX, CY, TAG_SIZE, DESIRED_SIZE, LAMBDA_P, PATCH, Z_DES, P_CB_X, P_CB_Y, P_CB_Z, MAX_LIN_VEL, MAX_ANG_VEL


# Camera-to-Body Rotation Matrix (From your base code)
R_CB = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=float)

p_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])

class IBVS_Control(Node):
    def __init__(self):
        super().__init__("IBVS_Control")
        self.bridge = CvBridge()

        # Subscriptions
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.cb_depth, 10)

        # Publisher - Sending velocity setpoints to MAVROS
        self.vel_pub = self.create_publisher(Twist, "/mavros/setpoint_velocity/cmd_vel_unstamped", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None

        # Define the desired image features (4 corners)
        s = DESIRED_SIZE // 2
        self.desired_pts = np.array([
            [CX - s, CY - s], # TL
            [CX + s, CY - s], # TR
            [CX + s, CY + s], # BR
            [CX - s, CY + s]  # BL
        ], dtype=np.float32)

        self.p_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])

        self.get_logger().info("IBVS Controller Node Started")

    def cb_depth(self, msg):
        # RealSense 16UC1 is millimeters, convert to meters
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) * 0.001

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,  x*y,      -(1+x*x),  y],
            [0,    -1/Z,  y/Z,  1+y*y,    -x*y,     -x],
            [0,     0,   -1,   -y*Z,       x*Z,      0]
        ])

    def cb_corners(self, msg):
        if self.depth_img is None:
            return

        pts = np.array([[p.x, p.y] for p in msg.polygon.points])
        rows, errs = [], []
        h, w = self.depth_img.shape

        # 1. Build Interaction Matrix and Error Vector
        for i in range(4):
            u, v = pts[i]
            ui, vi = int(round(u)), int(round(v))

            if not (0 <= ui < w and 0 <= vi < h): return

            # Depth sampling with safety patch
            patch = self.depth_img[max(0, vi-PATCH):min(h, vi+PATCH+1),
                                   max(0, ui-PATCH):min(w, ui+PATCH+1)]
            valid = patch[patch > 0]
            if valid.size == 0:
                self.get_logger().warn(f"No valid depth for point {i+1}")
                return

            Z = float(np.median(valid))

            L = self.interaction_matrix(u, v, Z)
            rows.append(L)

            # Normalize the current point and the desired point
            curr_x = (u - CX) / FX
            curr_y = (v - CY) / FY
            des_x = (self.desired_pts[i, 0] - CX) / FX
            des_y = (self.desired_pts[i, 1] - CY) / FY
            
            # Error in normalized coordinates
            errs.extend([curr_x - des_x, curr_y - des_y, Z - Z_DES])

        Ls = np.vstack(rows)
        e = np.array(errs).reshape(-1, 1)

        # --- TERMINAL PRINTING: FEATURE ERRORS ---
        print("\n" + "="*50)
        print(f"{'Feature':<10} | {'du (px)':<10} | {'dv (px)':<10} | {'dz (m)':<10}")
        print("-" * 50)

        # errs contains [u1, v1, z1, u2, v2, z2, ...]
        for i in range(4):
            ex = errs[i*3]     # u error
            ey = errs[i*3 + 1] # v error
            ez = errs[i*3 + 2] # depth error
            print(f"Point {i+1:<4} | {ex:+10.1f} | {ey:+10.1f} | {ez:+10.3f}")

        print("-" * 50)

        # 2. Compute Camera Velocity (Vc)
        Vc = -LAMBDA_P * (np.linalg.pinv(Ls) @ e)

        # Split into linear and angular components
        v_c = Vc[0:3].reshape(3, 1)
        w_c = Vc[3:6].reshape(3, 1)

        # 3. Transform to Body Frame (Vb)
        # Rotate angular velocity: w_b = R_CB * w_c
        w_b = R_CB @ w_c

        # Rotate linear velocity and compensate for lever arm (p_CB)
        # v_b = R_CB * v_c + (w_b x p_CB)
        v_b = (R_CB @ v_c) + np.cross(w_b.flatten(), p_CB).reshape(3, 1)

        # 4. Apply Velocity Capping and Type Casting
        # We use .item() to extract the raw value and float() to ensure ROS2 compatibility.
        # BlueROV2 / ArduSub Standard: 
        # x: Forward, y: Lateral (Strafe), z: Vertical (Throttle)
        vx = float(np.clip(v_b[0].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vy = float(np.clip(v_b[1].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vz = float(np.clip(v_b[2].item(), -MAX_LIN_VEL, MAX_LIN_VEL))

        # Angular: 
        # x: Roll, y: Pitch, z: Yaw
        wx = float(np.clip(w_b[0].item(), -MAX_ANG_VEL, MAX_ANG_VEL))
        wy = 0.0  # Keep pitch at 0 to maintain ROV stability unless needed
        wz = float(np.clip(w_b[2].item(), -MAX_ANG_VEL, MAX_ANG_VEL))

        # 5. Build and Publish Twist Message
        cmd = Twist()
        
        # Linear Velocities
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.linear.z = vz

        # Angular Velocities
        cmd.angular.x = wx
        cmd.angular.y = wy
        cmd.angular.z = wz

        # Safety Check: Log the command to terminal
        self.get_logger().info(f"Vx: {vx:.2f}, Vy: {vy:.2f}, Vz: {vz:.2f}, Wx: {wx:.2f}, Wy: {wy:.2f}, Wz: {wz:.2f}")
        self.vel_pub.publish(cmd)

        errs_np = np.array(errs, dtype=np.float32).reshape(4, 3)
        err_msg = Float32MultiArray()
        
        # Flatten in the order:
        # [eu1, ev1, ez1, eu2, ev2, ez2, eu3, ev3, ez3, eu4, ev4, ez4]
        err_msg.data = errs_np.flatten().tolist()
        self.err_pub.publish(err_msg)

def main():
    rclpy.init()
    node = IBVS_Control()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
