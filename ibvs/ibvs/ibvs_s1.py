#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from mavros_msgs.msg import OverrideRCIn

from ibvs.constants import *
from ibvs.dynamics import *

R_CB = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
p_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        # --- Subscriber
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.cb_depth, 10)

        # --- Publisher
        self.rc_pub = self.create_publisher(OverrideRCIn, "/mavros/rc/override", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None

        self.desired_pts = self.desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE)

        self.M = build_M_matrix(
            mass=15.0,
            add_frac_x=0.2,
            add_frac_y=0.3,
            add_frac_z=0.4,
            L=0.46,
            W=0.40,
            H=0.25,
        )
        self.D = build_D_matrix(0.2)

        self.Fz_bias = -0.3  # sinking compensation (tunable)

    def force_to_pwm(self, u, u_max, pwm_center=1500, pwm_range=400):
        u = np.clip(u, -u_max, u_max)
        return int(pwm_center + (u / u_max) * pwm_range)

    def desired_corners(self, Z_DES , fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners = np.array([
            [-half, -half, Z_DES],
            [ half, -half, Z_DES],
            [ half,  half, Z_DES],
            [-half,  half, Z_DES],
        ])

        pts = np.zeros((4, 2), dtype=float)
    
        for i, (X, Y, Z) in enumerate(corners):
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            pts[i] = [u, v]
        return pts

    def cb_depth(self, msg):
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

        rows,errs=[],[]
        pts = np.array([[p.x,p.y] for p in msg.polygon.points])
        h,w = self.depth_img.shape

        for i,(u,v) in enumerate(pts):
            ui,vi=int(u),int(v)
            if not (0 <= ui < w and 0 <= vi < h):
                return
            
            patch = self.depth_img[
                max(0, vi-PATCH):min(h, vi+PATCH+1),
                max(0, ui-PATCH):min(w, ui+PATCH+1)]
            valid = patch[patch > 0]
            if valid.size == 0:
                return
                
            Z = float(np.median(valid))
            if Z < 0.2:
                return
                
            rows.append(self.interaction_matrix(u, v, Z))
            
            x, y = (u - CX)/ FX, (v - CY)/ FY
            xd, yd = (self.desired_pts[i]-[CX,CY])/[FX,FY]
            errs.extend([x - xd, y - yd, Z - Z_DES])

        Ls = np.vstack(rows)
        e = np.array(errs).reshape(-1,1)

        Vc = -LAMBDA_P * np.linalg.pinv(Ls) @ e

        v_c = Vc[0:3].reshape(3, 1)
        w_c = Vc[3:6].reshape(3, 1)
        
        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        nu = np.zeros(6)
        nu[0:3] = Vb.flatten()
        nu[3:6] = Wb.flatten()

        # Linearized acceleration estimate
        nu_dot = nu / 1.0

        # Gravity / buoyancy
        g = np.array([0,0,self.Fz_bias,0,0,0])

        # Tau computation
        Tau = self.M @ nu_dot + self.D @ nu + g

        # RC OVERRIDE
        rc = OverrideRCIn()
        rc.channels = [65535]*18

        rc.channels[2] = self.force_to_pwm(Tau[2],  MAX_TAU_Z)    # Heave
        rc.channels[3] = self.force_to_pwm(Tau[5],  MAX_TAU_YAW)  # Yaw
        rc.channels[4] = self.force_to_pwm(Tau[0],  MAX_TAU_X)    # Surge
        rc.channels[5] = self.force_to_pwm(Tau[1],  MAX_TAU_Y)    # Sway

        self.rc_pub.publish(rc)

        # Publish error
        err_msg = Float32MultiArray()
        err_msg.data = np.array(errs, dtype=np.float32).tolist()
        self.err_pub.publish(err_msg)

def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
