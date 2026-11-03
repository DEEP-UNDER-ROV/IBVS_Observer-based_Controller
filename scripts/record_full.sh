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
    /ibvs/error/px \
    /ibvs/error/no \
    /ibvs/ukf/data \
    /ibvs/nu_B_hat \
    /ibvs/pos_hat \
    /ibvs/torque \
    /mavros/imu/data \
    /mavros/imu/data_raw \
    /camera/camera/accel/sample \
    /camera/camera/gyro/sample \
    /camera/camera/imu \
    /mavros/rc/override \
    /camera/overlay/image_raw/compressed