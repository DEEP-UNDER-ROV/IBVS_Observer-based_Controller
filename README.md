# Stereo-enhanced Image-based Visual Servoing (IBVS) for U-ROV with Observer-based Controller

A ROS2 implementation of **Stereo-enhanced Image-based Visual Servoing (IBVS)** for Underwater ROV with Observer-based controller using **stereo vision camera**, **AprilTag feature extraction**, **pixhawk IMU sensor**, and a **PD-augmented control law** via ssh communication between Ground Station and the U-ROV.

<img width="400" height="235" alt="frame_proj" src="https://github.com/user-attachments/assets/735eaa02-1c44-456e-be8e-baf356a61081" /> 
<img width="400" height="235" alt="image" src="https://github.com/user-attachments/assets/272a90fc-369b-4f92-8113-ba26e3a0ce3c" />





The system estimates target pose directly from image features and generates PWM commands sent to the FCU via MAVRROS for closed-loop visual servoing.

---

## System Architecture
<img width="1813" height="432" alt="image" src="https://github.com/user-attachments/assets/9a4b908b-dc3b-4f65-9108-72da0144fe0c" />

## Dependencies

* ROS2 Jazzy
* Mavlink MAVROS
* RealSense library
* Python 3.10+
* OpenCV
* QGround Control
* DU Perception AprilTag Triangulate Library <br>
  https://github.com/DEEP-UNDER-ROV/DU_Perception_Apriltag_Triangulate

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

The IBVS node have several parameters to be configured:
* matrix_3d    -> Configure the interaction matrix to use 3D or 2D matrix
* matrix_delta -> Configure using depth or delta variable to represent the distance to tag
* control_dls  -> Configure the IBVS control interaction matrix computation
* stereo_cam   -> Configure to use both left and right IR cam (Stereo-IBVS)
* ukf_fossen   -> Configure the UKF model to use Fossen dynamics instead of IMU model

To run the respective parameters use
```bash
ros2 run ibvs ibvs --ros-args -p (parameter):=True/False
```
---

## Data record for Analyze
To record the necessary topics for data analyzation, this scripts can be run as follows:
Minimalize record (for later code debug):
```bash
bash scripts/record_min.sh
```
or
```bash
ros2 launch ibvs record_min.launch.py
```

For full record to plot the controller data:
```bash
bash scripts/record_full.sh
```
or
```bash
ros2 launch ibvs record_full.launch.py
```


---

## Published Topics

| Topic                 | Description               |
| --------------------- | ------------------------  |
| `/apriltag/corners`   | AprilTag position corner features  |
| `/detection1`         | Left IR camera pixel detection   |
| `/detection2`         | Right IR ccamera pixel detection   |
| `/ibvs/error/px`      | Pixel image feature error |
| `/ibvs/error/no`      | Normalize image feature error |
| `/ibvs/nu_B_hat`      | Estimated body velocity   |
| `/ibvs/pos_hat`       | Estimated position in FLU frame   |
| `/ibvs/ukf/data`      | UKF data debug            |
| `/mavros/rc/override` | RC override commands      |
| `/camera/overlay`     | GCS Stream Visualizaiton  |
