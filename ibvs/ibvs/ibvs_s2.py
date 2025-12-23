#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import PolygonStamped, Twist, Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ibvs.constants import *

R_CB = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=float)
p_CB = np.array([P_CB_X, P_CB_Y, P_CB_Z])

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
        self.desired_pts = self._make_desired_pts()

    def desired_corners(self, Z_DES , fx, fy, cx, cy, tag_size):
        half = tag_size / 2.0
        corners_3d = np.array([
            [-half, -half, Z_DES],
            [ half, -half, Z_DES],
            [ half,  half, Z_DES],
            [-half,  half, Z_DES],
        ])

        desired = np.zeros((4, 2), dtype=float)
    
        for i, (X, Y, Z) in enumerate(corners_3d):
            u = fx * (X / Z) + cx
            v = fy * (Y / Z) + cy
            desired[i] = [u, v]
    
        return desired

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

        self.desired_pts = self.desired_corners(Z_DES, FX, FY, CX, CY, TAG_SIZE)

        rows,errs=[],[]
        pts = np.array([[p.x,p.y] for p in msg.polygon.points])
        h,w = self.depth_img.shape

        for i,(u,v) in enumerate(pts):
            ui,vi=int(u),int(v)
            
            Z = np.median(self.depth_img[
                max(0, vi-PATCH):vi+PATCH+1,
                max(0, ui-PATCH):ui+PATCH+1
                ])
            
            if Z <= 0 or np.isnan(Z):
                return
                
            rows.append(self.interaction_matrix(u,v,Z))
            x,y = (u-CX)/FX, (v-CY)/FY
            ud, vd = self.desired_pts[i]
            xd,yd = (ud - CX) / FX, (vd - CY) / FY
            errs.extend([x - xd, y - yd, Z - Z_DES])

        L = np.vstack(rows)                    # (12x6)
        e = np.array(errs).reshape(-1, 1)      # (12x1)

        Vc = -LAMBDA_P * np.linalg.pinv(L) @ e
        Vb = R_CB @ Vc[0:3]

        self.p_cmd += Vb.flatten() * dt
        self.p_cmd = np.clip(self.p_cmd, -MAX_OFFSET, MAX_OFFSET)

        vel=Twist()
        vel.linear.x,vel.linear.y,vel.linear.z=Vb.flatten()
        self.vel_pub.publish(vel)

        self.pos_pub.publish(Point(x = self.p_cmd[0],y = self.p_cmd[1],z = self.p_cmd[2]))
        self.err_pub.publish(Float32MultiArray(data=np.array(errs,dtype=np.float32).tolist()))

def main():
    rclpy.init()
    rclpy.spin(IBVSControllerNode())
    rclpy.shutdown()

if __name__=="__main__":
    main()
