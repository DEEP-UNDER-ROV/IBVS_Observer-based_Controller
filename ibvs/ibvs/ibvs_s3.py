#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3Stamped

from ibvs.constants import *

R_CB = np.array([[0,0,1],[1,0,0],[0,1,0]])

class IBVSTvecNode(Node):
    def __init__(self):
        super().__init__("ibvs_tvec_node")
        self.sub_tvec = self.create_subscription(
            Vector3Stamped, "/pnp/tvec", self.cb_tvec, 10)

        self.pos_pub = self.create_publisher(Point, "/ibvs/pos", 10)
        self.err_pub = self.create_publisher(Float32MultiArray, "/ibvs/error", 10)

        self.tvec_des = np.array([0,0,Z_DES])

    def cb_tvec(self,msg):
        tvec=np.array([msg.vector.x,msg.vector.y,msg.vector.z])
        e=tvec-self.tvec_des
        e_body=R_CB@e

        p_cmd=-K_P*e_body
        self.pos_pub.publish(Point(x=p_cmd[0],y=p_cmd[1],z=p_cmd[2]))
        self.err_pub.publish(Float32MultiArray(data=e.astype(np.float32).tolist()))

def main():
    rclpy.init()
    rclpy.spin(IBVSTvecNode())
    rclpy.shutdown()

if __name__=="__main__":
    main()
