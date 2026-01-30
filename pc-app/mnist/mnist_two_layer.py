import numpy as np
import logging
from keras.utils import to_categorical
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from cnn.network import TwolayerNet


# ==========================================
# 1. データの準備 (Keras使用)
# ==========================================
print("Loading MNIST data...")
(x_train, t_train), (x_test, t_test) = load_data()

# 前処理: (N, 28, 28) -> (N, 784) に平坦化し、0.0~1.0に正規化
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 正解ラベルをOne-Hot表現に変換 (例: 5 -> [0,0,0,0,0,1,0...])
t_train = to_categorical(t_train, 10)
t_test = to_categorical(t_test, 10)

print(f"Train Data: {x_train.shape}")
print(f"Train Label: {t_train.shape}")

# ==========================================
# 2. ネットワークの構築
# ==========================================
network = TwolayerNet(input_size=784, hidden_size=50, output_size=10)

# ハイパーパラメータ
iters_num = 10000  # 繰り返す回数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
print("\nStart training...")

# ==========================================
# 3. 学習ループ
# ==========================================
for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算 (ここが自作ライブラリの処理)
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータの更新 (SGD)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 記録 (1000回に1回だけ表示)
    if i % 1000 == 0:
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print(f"Iter {i:5d} | Loss: {loss:.4f}")

print("Training finished!")

# ==========================================
# 4. 精度の評価
# ==========================================
# テストデータを使って正解率を計算
correct_count = 0
# メモリ節約のためバッチごとに計算
for i in range(len(x_test) // batch_size):
    x_batch = x_test[i*batch_size : (i+1)*batch_size]
    t_batch = t_test[i*batch_size : (i+1)*batch_size]
    
    y = network.predict(x_batch)
    
    # 最大値のインデックス(予測した数字)を取得
    p = np.argmax(y, axis=1)
    t = np.argmax(t_batch, axis=1)
    
    correct_count += np.sum(p == t)

accuracy = correct_count / len(x_test)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

# ... (学習ループと精度の表示が終わった後) ...

# ==========================================
# 5. 結果の可視化 (Pickle不要！メモリ上のnetworkをそのまま使う)
# ==========================================
print("\nVisualizing results...")

# テストデータからランダムに20枚選ぶ
indices = np.random.choice(len(x_test), 20, replace=False)

fig = plt.figure(figsize=(12, 10))
fig.suptitle("Model Predictions (Blue=Correct, Red=Wrong)", fontsize=16)

for i, idx in enumerate(indices):
    # 推論実行 (networkは学習済みの状態)
    img_data = x_test[idx].reshape(1, 784) # (1, 784)に変形
    y = network.predict(img_data)
    pred_label = np.argmax(y, axis=1)[0]
    
    # 正解ラベル (One-Hotから数字に戻す)
    true_label = np.argmax(t_test[idx])
    
    # プロット
    ax = fig.add_subplot(4, 5, i+1)
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray') # 元の画像サイズに戻して表示
    ax.axis('off')
    
    color = 'blue' if pred_label == true_label else 'red'
    ax.set_title(f"Pred: {pred_label}\n(Ans: {true_label})", color=color, fontsize=14)

plt.tight_layout()
plt.show()
