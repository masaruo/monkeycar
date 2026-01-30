import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from loader import Loader
from cnn.network import DeepConvNet

def load_weights(network, weights_dir="weights_bin"):
    """バイナリファイルから重みを読み込む"""
    print(f"Loading weights from {weights_dir}...")
    
    if not os.path.exists(weights_dir):
        print(f"Error: {weights_dir} not found. Train the model first.")
        sys.exit(1)

    loaded_count = 0
    for key in network.params.keys():
        file_path = os.path.join(weights_dir, f"{key}.bin")
        if os.path.exists(file_path):
            # float32として読み込み、元の形状にreshapeする
            data = np.fromfile(file_path, dtype=np.float32)
            try:
                network.params[key] = data.reshape(network.params[key].shape)
                loaded_count += 1
            except ValueError:
                print(f"Error: Shape mismatch for {key}. Expected {network.params[key].shape}, got {data.shape}")
        else:
            print(f"Warning: {file_path} not found.")

    # 重みをレイヤーに反映
    # (SimpleConvNet/DeepConvNetの設計によっては、params更新後にレイヤの再生成が必要な場合があるが
    #  今回の実装ではレイヤが self.params の参照を持っているか、都度渡す形であればOK。
    #  念のため、手動でセットする)
    
    # Conv1
    network.layers['Conv1'].W = network.params['W1']
    network.layers['Conv1'].b = network.params['b1']
    # Conv2
    if 'Conv2' in network.layers:
        network.layers['Conv2'].W = network.params['W2']
        network.layers['Conv2'].b = network.params['b2']
    
    # Affine layers (名前が動的かもしれないので決め打ちで対応)
    if 'Affine1' in network.layers:
        # DeepConvNetの場合、W3/b3がAffine1
        if 'W3' in network.params:
            network.layers['Affine1'].W = network.params['W3']
            network.layers['Affine1'].b = network.params['b3']
        # SimpleConvNetの場合、W2/b2がAffine1
        elif 'W2' in network.params:
            network.layers['Affine1'].W = network.params['W2']
            network.layers['Affine1'].b = network.params['b2']

    if 'Affine2' in network.layers:
         # DeepConvNetの場合、W4/b4がAffine2
        if 'W4' in network.params:
            network.layers['Affine2'].W = network.params['W4']
            network.layers['Affine2'].b = network.params['b4']
        # SimpleConvNetの場合、W3/b3がAffine2
        elif 'W3' in network.params:
            network.layers['Affine2'].W = network.params['W3']
            network.layers['Affine2'].b = network.params['b3']

    print(f"Loaded {loaded_count} parameter files.")

def draw_steering(img, steering, color=(0, 255, 0), length=40, thickness=2):
    """画像中心からステアリング方向へ線を描画"""
    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h
    
    # ステアリング -1.0(左)~1.0(右) を角度に変換
    # ここでは -1.0 -> -45度, 1.0 -> 45度 と仮定
    max_angle = np.pi / 4  # 45度
    angle = steering * max_angle
    
    end_x = int(center_x + length * np.sin(angle))
    end_y = int(center_y - length * np.cos(angle))
    
    cv2.line(img, (center_x, center_y), (end_x, end_y), color, thickness)
    return img

def main():
    # 1. データの準備 (Trainerと同じ設定で)
    loader = Loader(data_dir="./data", image_size=(160, 120))
    images, steerings, throttles, _ = loader.load_sessions()
    
    if len(images) == 0:
        print("No data found.")
        return

    # 前処理
    if images.ndim == 4 and images.shape[3] == 3:
        x_all = images.transpose(0, 3, 1, 2) # (N, C, H, W)
    else:
        x_all = images
    x_all = x_all.astype('float32') / 255.0
    t_all = np.column_stack((steerings, throttles)).astype('float32')

    # テストデータのみ抽出 (後半20%)
    split_idx = int(len(x_all) * 0.8)
    x_test = x_all[split_idx:]
    t_test = t_all[split_idx:]
    
    print(f"Test data size: {len(x_test)}")

    # 2. ネットワークの構築 (DeepConvNet)
    # ★重要: trainer.py と同じパラメータにすること！
    input_dim = x_test.shape[1:]
    network = DeepConvNet(
        input_dim=input_dim,
        conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
        conv_param_2={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
        hidden_size=100,
        output_size=2
    )

    # 3. 重みのロード
    load_weights(network, weights_dir="weights_bin")

    # 4. 可視化
    samples = 15
    indices = np.random.choice(len(x_test), samples, replace=False)
    
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Green: Human (Truth) / Red: AI (Prediction)", fontsize=16)

    for i, idx in enumerate(indices):
        # 画像 (表示用に H,W,C に戻す)
        img_data = x_test[idx] # (3, 120, 160)
        img_disp = img_data.transpose(1, 2, 0).copy() # (120, 160, 3)
        img_disp = (img_disp * 255).astype(np.uint8)
        img_disp = np.ascontiguousarray(img_disp)

        # 推論
        x_in = img_data.reshape(1, *input_dim)
        pred = network.predict(x_in)[0] # [steer, throttle]
        true = t_test[idx]

        # 描画
        # 緑: 正解
        draw_steering(img_disp, true[0], color=(0, 255, 0), length=50, thickness=3)
        # 赤: 予測
        draw_steering(img_disp, pred[0], color=(255, 0, 0), length=50, thickness=2)

        # サブプロット
        ax = fig.add_subplot(3, 5, i+1)
        ax.imshow(img_disp)
        ax.axis('off')
        
        # タイトル (スロットル値比較)
        diff = abs(true[1] - pred[1])
        title_color = 'black' if diff < 0.2 else 'red'
        ax.set_title(f"Thr: {true[1]:.2f} / {pred[1]:.2f}", color=title_color, fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
