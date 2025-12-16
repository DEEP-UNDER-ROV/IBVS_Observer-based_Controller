#!/usr/bin/env python3
import rclpy
from rclpy.node import Node



from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool



import pyrealsense2 as rs
import cv2
import numpy as np
import math
from pupil_apriltags import Detector



# ================================================================
#               === YOUR ORIGINAL IBVS PARAMETERS ===
# ================================================================
LAMBDA_P = 0.05
LAMBDA_D = 0.02
DT = 0.01



TARGET_TAG_ID = 1
DESIRED_SIZE = 180
PATCH = 2
EPS = 1e-6



Z_des = 0.40
depth_tol = 0.03
Kp_z = 20
K_ROT = 1.5



desired_pts = np.array([
    [320-DESIRED_SIZE//2,240-DESIRED_SIZE//2],
    [320+DESIRED_SIZE//2,240-DESIRED_SIZE//2],
    [320+DESIRED_SIZE//2,240+DESIRED_SIZE//2],
    [320-DESIRED_SIZE//2,240+DESIRED_SIZE//2]
], dtype=float)



# =====================================================================
#                  ===  INTERACTION MATRIX  ===
# =====================================================================
def interaction_matrix(u, v, Z, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy
    return np.array([
        [-1.0/Z, 0.0, x/Z, x*y, -(1 + x*x), y],
        [0.0, -1.0/Z, y/Z, 1 + y*y, -x*y, -x]
    ])



def build_IBVS_matrix(pts, desired, depth_img, fx, fy, cx, cy):
    rows, errs = [], []
    h, w = depth_img.shape



    for i in range(4):
        u, v = pts[i]
        ui, vi = int(round(u)), int(round(v))



        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            return None, None, False, None



        patch = depth_img[max(0,vi-PATCH):min(h,vi+PATCH+1),
                          max(0,ui-PATCH):min(w,ui+PATCH+1)]
        valid = patch[patch > 0]



        if valid.size == 0:
            return None, None, False, None



        Z = float(np.median(valid))
        if Z <= 0 or np.isnan(Z):
            return None, None, False, None



        L = interaction_matrix(u, v, Z, fx, fy, cx, cy)
        rows.extend([L[0], L[1]])
        errs.append([u - desired[i,0]])
        errs.append([v - desired[i,1]])



    return np.vstack(rows), np.vstack(errs), True, Z



# =====================================================================
#                     === ROS2 IBVS OFFBOARD NODE ===
# =====================================================================
class IBVSOffboardNode(Node):



    def __init__(self):
        super().__init__("ibvs_offboard_controller")



        # MAVROS publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            "/mavros/setpoint_velocity/cmd_vel_unstamped",
            10
        )



        # MAVROS state
        self.state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self.state_cb,
            10
        )



        # Services
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.setmode_client = self.create_client(SetMode, "/mavros/set_mode")



        self.current_state = State()
        self.timer = self.create_timer(0.05, self.keep_alive)



        # IBVS Realsense initializers
        self.pipeline = None
        self.depth_scale = None
        self.prev_e = None



        self.start_realsense()
        self.detector = Detector(families="tag36h11", nthreads=4)



        self.get_logger().info("IBVS OFFBOARD Node initialized.")



    # ----------------------- PX4 state callback -----------------------
    def state_cb(self, msg):
        self.current_state = msg



    # ---------------------- Keep OFFBOARD alive -----------------------
    def keep_alive(self):
        """
        Even if IBVS loses target, PX4 must continuously receive velocity
        setpoints or it will exit OFFBOARD.
        """
        if self.current_state.mode == "OFFBOARD":
            msg = Twist()
            self.cmd_vel_pub.publish(msg)



    # ---------------------- Start Realsense ---------------------------
    def start_realsense(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)



        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()



        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole = rs.hole_filling_filter()



        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = color_intr.fx, color_intr.fy
        self.cx, self.cy = color_intr.ppx, color_intr.ppy



        self.pipeline = pipeline
        self.get_logger().info("Realsense started with intrinsics loaded.")



    # --------------------------- MAIN LOOP -----------------------------
    def run_ibvs(self):
        self.get_logger().info("Starting IBVS loop...")



        while rclpy.ok():



            # Ensure OFFBOARD mode
            self.activate_offboard_mode()



            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)



            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()



            # Post-processing depth
            depth_frame = self.spatial.process(depth_frame)
            depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole.process(depth_frame)



            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
            color = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)



            results = self.detector.detect(gray)
            twist_msg = Twist()



            if len(results) > 0:
                det = results[0]   # take the closest tag
                if det.tag_id == TARGET_TAG_ID:
                    pts = det.corners.astype(float)
                    Ls, e, ok, Z_current = build_IBVS_matrix(
                        pts, desired_pts, depth, self.fx, self.fy, self.cx, self.cy
                    )



                    if ok:
                        if self.prev_e is None:
                            e_dot = np.zeros_like(e)
                        else:
                            e_dot = (e - self.prev_e) / DT
                        self.prev_e = e.copy()



                        Lpinv = np.linalg.pinv(Ls)
                        Vc = -(LAMBDA_P * (Lpinv @ e) + LAMBDA_D * (Lpinv @ e_dot))
                        Vc = Vc.flatten()



                        # Depth regulation
                        error_z = Z_current - Z_des
                        if abs(error_z) < depth_tol:
                            error_z = 0.0
                        Vc[2] = Kp_z * error_z  # body z control



                        # Camera → body transform
                        R = np.array([
                            [0,0,1],
                            [1,0,0],
                            [0,1,0]
                        ], float)



                        v_body = R @ Vc[0:3]
                        w_body = R @ Vc[3:6]



                        # Orientation correction
                        p1, p2 = pts[2], pts[3]
                        theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
                        theta = (theta + np.pi) % (2*np.pi) - np.pi
                        omega_x_cmd = -K_ROT * theta



                        # ---------- FINAL TWIST ----------
                        twist_msg.linear.x  = float(v_body[0])
                        twist_msg.linear.y  = float(v_body[1])
                        twist_msg.linear.z  = float(v_body[2])
                        twist_msg.angular.x = float(omega_x_cmd)
                        twist_msg.angular.y = float(w_body[1])
                        twist_msg.angular.z = float(w_body[2])



            # Publish IBVS or zero velocity
            self.cmd_vel_pub.publish(twist_msg)



            # Show debug feed
            cv2.imshow("IBVS ROV", color)
            if cv2.waitKey(1) & 0xFF == 27:
                break



    # -------------------------- OFFBOARD MODE --------------------------
    def activate_offboard_mode(self):
        if self.current_state.mode != "OFFBOARD":
            req = SetMode.Request()
            req.custom_mode = "OFFBOARD"
            self.setmode_client.call_async(req)



        if not self.current_state.armed:
            arm = CommandBool.Request()
            arm.value = True
            self.arming_client.call_async(arm)





# =====================================================================
#                            MAIN
# =====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = IBVSOffboardNode()



    try:
        node.run_ibvs()
    except KeyboardInterrupt:
        pass



    rclpy.shutdown()





if __name__ == "__main__":
    main()
 
