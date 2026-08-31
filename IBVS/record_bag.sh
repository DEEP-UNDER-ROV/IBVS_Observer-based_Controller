#!/bin/bash

# Navigate to data directory
DATA_DIR="$HOME/rov_ws/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

# Dynamic timestamped folder name (e.g., rosbag_2026_08_31-20_30_00)
BAG_NAME="rosbag_$(date +'%Y_%m_%d-%H_%M_%S')"
QOS_PATH="$DATA_DIR/qos_overrides.yaml"

echo "Starting ROS 2 bag recording: $BAG_NAME..."
echo "Using QoS Overrides: $QOS_PATH"

ros2 bag record -s mcap \
  -o "$BAG_NAME" \
  --max-cache-size 100000000 \
  --qos-profile-overrides-path "$QOS_PATH" \
  /mavros/imu/data \
  /mavros/rc/override \
  /ibvs/ukf/data \
  /ibvs/nu_B_hat \
  /ibvs/torque \
  /ibvs/error/px \
  /ibvs/error/no \
  /apriltag/corners \
  /detection1 \
  /detection2 \
  /camera/overlay/image_raw/compressed
