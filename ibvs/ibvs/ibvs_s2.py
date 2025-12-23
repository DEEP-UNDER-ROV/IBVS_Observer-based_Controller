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

class IBVSIntegratedNode(Node):
    def __init__(self):
        super().__init__("ibvs_integrated_node")
        self.bridge = CvBridge()
        self.sub_corners = self.create_subscription(
            PolygonStamped, "/apriltag/corners", self.cb_corners, 10)
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw", self.cb_depth, 10)

        self.vel_pub = self.create_publisher(Twist, "/ibvs/vel", 10)
        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.depth_img = None
        self.p_cmd = np.zeros(3)
        self.last_time = self.get_clock().now()
        self.desired_pts = self._make_desired_pts()

    def _make_desired_pts(self):
        s = DESIRED_SIZE // 2
        return np.array([[CX-s,CY-s],[CX+s,CY-s],[CX+s,CY+s],[CX-s,CY+s]])

    def cb_depth(self,msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg).astype(np.float32)*0.001

    def interaction_matrix(self,u,v,Z):
        x,y=(u-CX)/FX,(v-CY)/FY
        return np.array([
            [-1/Z,0,x/Z,x*y,-(1+x*x),y],
            [0,-1/Z,y/Z,1+y*y,-x*y,-x],
            [0,0,-1,-y*Z,x*Z,0]
        ])

    def cb_corners(self,msg):
        if self.depth_img is None: return
        now = self.get_clock().now()
        dt = (now-self.last_time).nanoseconds*1e-9
        self.last_time = now
        if dt<=0: return

        rows,errs=[],[]
        pts=np.array([[p.x,p.y] for p in msg.polygon.points])
        h,w=self.depth_img.shape

        for i,(u,v) in enumerate(pts):
            ui,vi=int(u),int(v)
            Z=np.median(self.depth_img[vi-PATCH:vi+PATCH+1,ui-PATCH:ui+PATCH+1])
            rows.append(self.interaction_matrix(u,v,Z))
            cx,cy=(u-CX)/FX,(v-CY)/FY
            dx,dy=(self.desired_pts[i]-[CX,CY])/[FX,FY]
            errs.extend([cx-dx,cy-dy,Z-Z_DES])

        Vc=-LAMBDA_P*np.linalg.pinv(np.vstack(rows))@np.array(errs).reshape(-1,1)
        v_b=R_CB@Vc[0:3]

        self.p_cmd += v_b.flatten()*dt
        self.p_cmd = np.clip(self.p_cmd,-MAX_OFFSET,MAX_OFFSET)

        vel=Twist()
        vel.linear.x,vel.linear.y,vel.linear.z=v_b.flatten()
        self.vel_pub.publish(vel)

        self.pos_pub.publish(Point(x=self.p_cmd[0],y=self.p_cmd[1],z=self.p_cmd[2]))
        self.err_pub.publish(Float32MultiArray(data=np.array(errs,dtype=np.float32).tolist()))

def main():
    rclpy.init()
    rclpy.spin(IBVSIntegratedNode())
    rclpy.shutdown()

if __name__=="__main__":
    main()
