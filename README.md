# Stereo-enhanced Image-based Visual Servoing (IBVS) for Underwater ROV

A ROS2 implementation of **Stereo-enhanced Image-based Visual Servoing (IBVS)** for underwater ROV positioning using **stereo vision**, **AprilTag feature extraction**, and a **PI-augmented control law** via ssh communication between Ground Station and the U-ROV.

<img width="400" height="235" alt="frame_proj" src="https://github.com/user-attachments/assets/722671fe-7fa3-4187-a49f-8d7075927780" /> <img width="400" height="235" alt="U-ROVs2" src="https://github.com/user-attachments/assets/c78c6581-40a1-4368-84cd-c6bfb52926f0" />




The system estimates target pose directly from image features and generates PWM commands sent to the FCU via MAVRROS for closed-loop visual servoing.

---

## System Architecture

```
Stereo Camera InfraRed Left-right
       │
       ▼
AprilTag Detection Corner
       │
       ▼
Feature Extraction per each corner (u,v,Z)
       │
       ▼
Image Error Computation
       │
       ▼
Interaction Matrix (Ls)
       │
       ▼
PI-Augmented IBVS Controller
       │
       ▼
Body Velocity Command
       │
       ▼
MAVROS RC Override
       │
       ▼
ROV Motion
```

## Dependencies

* ROS2 Jazzy
* Python 3.10+
* OpenCV
* DU Perception AprilTag Triangulate Library <br>
  https://github.com/DEEP-UNDER-ROV/DU_Perception_Apriltag_Triangulate
* cv_bridge
* MAVROS

---

## Installation
Install required library for GStreamer
```bash
sudo apt install -y python3-opencv \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-tools

```

Clone the DU Perception repository first into your ROS2 workspace, follow the guides on readme.md. <br>
Clone IBVS repository into your ROS2 workspace;

```bash
cd ~/your_ws/src

git clone https://github.com/DEEP-UNDER-ROV/IBVS.git
```

Build the package.

```bash
cd ~/your_ws

colcon build

source install/setup.bash
```

---

## Running
Launch DU Perception AprilTag Triangulate

```bash
ros2 launch rov_vision uw_rs_apriltag_triangulation.launch.xml
ros2 param set /camera/camera depth_module.emitter_enabled 0
```

Launch MAVROS node:

```bash
ros2 launch mavros apm.launch fcu_url:="serial:///dev/ttyACM0:921600" gcs_url:="udp:ssh_ip"
```


Launch IBVS system:

```bash
ros2 launch ibvs visualize
ros2 launch ibvs ibvs
```

---

## Published Topics

| Topic                 | Description              |
| --------------------- | ------------------------ |
| `/apriltag/corners`   | AprilTag corner features |
| `/ibvs/error`         | Image feature error      |
| `/ibvs/vel_body`      | Desired body velocity    |
| `/ibvs/pwm_debug`     | IBVS PWM Control         |
| `/mavros/rc/override` | RC override commands     |

---

## Controller Parameters

| Parameter      | Description              |
| -------------- | -------------------------|
| `Kp`           | Proportional gain        |
| `Ki`           | Integral gain            |
| `mu`           | Pseudoinverse Dampening  |

---
