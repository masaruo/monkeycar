#!/bin/bash
# ホストOS側で必要な設定
sudo apt update && sudo apt install -y python3-libcamera

# I2Cとカメラの有効化確認（config.txtへの反映は手動またはsedで）
echo "Please ensure 'camera_auto_detect=1' and 'dtparam=i2c_arm=on' are in /boot/firmware/config.txt"
