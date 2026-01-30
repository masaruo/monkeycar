#!/bin/bash

# var setting
PI_USER=team40
PI_HOST="team40.local"
PI_DATA_PATH="/home/team40/monkeycar/car-app/data/"
PI_MODEL_PATH="/home/team40/monkeycar/car-app/"
LOCAL_PATH_DATA="./data"
LOCAL_PATH_MODEL="./weights_bin"

# ラズパイから学習データをプル
pull_data(){
    echo "データをホストPCに移管中..."
    # a: archive, v: verbose, z: compress
    rsync -avz --progress ${PI_USER}@${PI_HOST}:${PI_DATA_PATH} ${LOCAL_PATH_DATA}
}

# ラズパイへ学習モデルをプッシュ
push_model(){
    echo "モデルを車に転送中"
    rsync -avz --progress ${LOCAL_PATH_MODEL} ${PI_USER}@${PI_HOST}:${PI_MODEL_PATH}
}

case "$1" in
    pull) pull_data ;;
    push) push_model ;;
    *) echo "Usage: $0 [pull|push]" ;;
esac
