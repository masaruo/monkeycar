import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets.mnist import load_data

# ★ここを変更: 新しい SimpleConvNet をインポート
from cnn.network import SimpleConvNet

# ==========================================
# 1. データの準備
# ==========================================
print("Loading MNIST data...")
(x_train, t_train), (x_test, t_test) = load_data()

# ★重要変更: CNN用に (N, 1, 28, 28) に変形します
# 以前は (-1, 784) でしたが、今回は画像の形状を維持します
x_train = x_train.reshape(-1, 1, 28, 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 1, 28, 28).astype('float32') / 255.0

t_train = to_categorical(t_train, 10)
t_test = to_categorical(t_test, 10)

print(f"Train Data: {x_train.shape}") # -> (60000, 1, 28, 28) になっているはず

# ==========================================
# 2. ネットワークの構築 (SimpleConvNet)
# ==========================================
# フィルター30枚、サイズ5x5、隠れ層100個
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# ハイパーパラメータ
iters_num = 5000  # CNNは計算が重いので少し減らしてもOK（十分賢くなります）
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
print("\nStart CNN training... (This may take a while on CPU)")

# ==========================================
# 3. 学習ループ
# ==========================================
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.gradient(x_batch, t_batch)
    
    # パラメータ更新 (W1, b1, W2, b2, W3, b3 全てを回す)
    for key in network.params.keys():
        network.params[key] -= learning_rate * grad[key]
    
    if i % 100 == 0:  # 頻繁に進捗を表示
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        print(f"Iter {i:5d} | Loss: {loss:.4f}")

print("Training finished!")

# ==========================================
# 4. 精度の評価
# ==========================================
print("Calculating accuracy...")
correct_count = 0
for i in range(len(x_test) // batch_size):
    x_batch = x_test[i*batch_size : (i+1)*batch_size]
    t_batch = t_test[i*batch_size : (i+1)*batch_size]
    
    y = network.predict(x_batch)
    p = np.argmax(y, axis=1)
    t = np.argmax(t_batch, axis=1)
    correct_count += np.sum(p == t)

accuracy = correct_count / len(x_test)
print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# 5. 結果の可視化
# ==========================================
print("\nVisualizing results...")
indices = np.random.choice(len(x_test), 20, replace=False)
fig = plt.figure(figsize=(12, 10))
fig.suptitle(f"CNN Predictions (Acc: {accuracy*100:.1f}%)", fontsize=16)

for i, idx in enumerate(indices):
    # 推論用: (1, 1, 28, 28)
    img_data = x_test[idx].reshape(1, 1, 28, 28)
    y = network.predict(img_data)
    pred_label = np.argmax(y, axis=1)[0]
    true_label = np.argmax(t_test[idx])
    
    ax = fig.add_subplot(4, 5, i+1)
    # 表示用: (28, 28)
    ax.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    ax.axis('off')
    
    color = 'blue' if pred_label == true_label else 'red'
    ax.set_title(f"Pred: {pred_label}\n(Ans: {true_label})", color=color, fontsize=14)

plt.tight_layout()
plt.show()
