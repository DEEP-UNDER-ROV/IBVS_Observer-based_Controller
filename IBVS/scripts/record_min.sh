#!/bin/bash

# Navigate to data directory
DATA_DIR="$HOME/rov_ws/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BAG_NAME="ibvs_${TIMESTAMP}"
echo "Starting rosbag recording: ${BAG_NAME}"

ros2 bag record \
    -o "${BAG_NAME}" \
    --max-cache-size 100000000 \
    --include-hidden-topics \
    /apriltag/corners \
    /detection1 \
    /detection2 \
    /mavros/imu/data \
    /mavros/imu/data_raw \
    /camera/camera/accel/sample \
    /camera/camera/gyro/sample \
    /camera/camera/imu \
    /corrected/left/image_raw/compressed