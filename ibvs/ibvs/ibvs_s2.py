#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PolygonStamped, Twist, Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *

class IBVSControllerNode(Node):
    def __init__(self):
        super().__init__("IBVSControllerNode")
        self.bridge = CvBridge()

        #Subscriber
        self.sub_corners = self.create_subscription(PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(Image, "/camera/depth/image_raw", self.cb_depth, 10)

        #Publisher
        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None
        self.p_cmd = np.zeros(3)
        self.last_time = self.get_clock().now()
        
        self.desired_pts = self.desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE)

        self.get_logger().info("CAUTION !! IBVS Control ON")

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

    def cb_depth(self,msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)*0.001

    def interaction_matrix(self, u, v, Z):
        x = (u - CX) / FX
        y = (v - CY) / FY
        return np.array([
            [-1/Z,  0,    x/Z,  x*y,      -(1+x*x),  y],
            [0,    -1/Z,  y/Z,  1+y*y,    -x*y,     -x],
            [0,     0,   -1,   -y*Z,       x*Z,      0]
        ])

    def cb_corners(self,msg):
        if self.depth_img is None: 
            return
            
        now = self.get_clock().now()
        dt = (now-self.last_time).nanoseconds*1e-9
        self.last_time = now
        if dt<=0: 
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

        L = np.vstack(rows)                    # (12x6)
        e = np.array(errs).reshape(-1, 1)      # (12x1)

        Vc = -LAMBDA_P * np.linalg.pinv(L) @ e
        
        v_c = Vc[0:3].reshape(3, 1)
        w_c = Vc[3:6].reshape(3, 1)

        Wb = R_CB @ w_c
        Vb = (R_CB @ v_c) + np.cross(Wb.flatten(), P_CB).reshape(3, 1)

        vel = Twist()
        vel.linear.x = float(np.clip(Vb[0].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.y = float(np.clip(Vb[1].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.linear.z = float(np.clip(Vb[2].item(), -MAX_LIN_VEL, MAX_LIN_VEL))
        vel.angular.x = float(np.clip(Wb[0].item(), -MAX_ANG_VEL, MAX_ANG_VEL))
        vel.angular.y = 0.0 
        vel.angular.z = float(np.clip(Wb[2].item(), -MAX_ANG_VEL, MAX_ANG_VEL))
        self.vel_pub.publish(vel)
        
        # Integrate velocity → position offset
        self.p_cmd += Vb.flatten() * dt
        self.p_cmd = np.clip(self.p_cmd, -MAX_OFFSET, MAX_OFFSET)
        self.p_cmd *= 0.995       
   
        self.pos_pub.publish(Point(
            x=float(self.p_cmd[0]),
            y=float(self.p_cmd[1]),
            z=float(self.p_cmd[2])
        ))

        errs_np = np.array(errs, dtype=np.float32).reshape(4, 3)
        err_msg = Float32MultiArray()
        # [eu1, ev1, ez1, eu2, ev2, ez2, eu3, ev3, ez3, eu4, ev4, ez4]
        err_msg.data = errs_np.flatten().tolist()
        self.err_pub.publish(err_msg)

def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()

if __name__=="__main__":
    main()
