import os
import numpy as np
from loader import Loader
from cnn.network import SimpleConvNet, DeepConvNet
from cnn.optimizer import Adam
from tqdm import trange

class Trainer:
    def __init__(self, data_dir="../data", output_dir="weights_bin", 
                 image_size=(160, 120), batch_size=32, learning_rate=0.01):
        """
        学習管理クラス
        
        Args:
            data_dir (str): 学習データ(sessionフォルダ)があるルートディレクトリ
            output_dir (str): 重みの保存先
            image_size (tuple): 画像サイズ (Width, Height)
            batch_size (int): ミニバッチサイズ
            learning_rate (float): 学習率
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        # self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate)
        
        # 内部状態
        self.network = None
        self.x_train = None
        self.t_train = None
        self.x_test = None
        self.t_test = None
        
        # データのロード
        self._prepare_data()
        
        # ネットワークの初期化
        self._init_network()

    def _prepare_data(self):
        """データの読み込みと前処理"""
        print(f"Loading data from {self.data_dir}...")
        loader = Loader(data_dir=self.data_dir, image_size=self.image_size)
        images, steerings, throttles, config = loader.load_sessions()
        
        if len(images) == 0:
            raise RuntimeError("No data found. Check your data path.")

        # 1. 画像の整形 (N, H, W, C) -> (N, C, H, W)
        # OpenCVなどは(H,W,C)だが、このCNN実装は(C,H,W)を期待するため
        if images.ndim == 4 and images.shape[3] == 3:
            images = images.transpose(0, 3, 1, 2)
            
        # 2. 正規化 (0.0 ~ 1.0)
        # Loader側ですでにされている場合もあるが、念のためfloat32変換と確認
        self.x_all = images.astype('float32') / 255.0
        
        # 3. 正解ラベルの結合 (Steering, Throttle)
        # どちらも回帰対象とする
        self.t_all = np.column_stack((steerings, throttles)).astype('float32')
        
        # 4. Train/Test分割 (8:2)
        data_size = self.x_all.shape[0]
        indices = np.random.permutation(data_size)
        split_idx = int(data_size * 0.8)
        
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        self.x_train = self.x_all[train_idx]
        self.x_test = self.x_all[test_idx]
        self.t_train = self.t_all[train_idx]
        self.t_test = self.t_all[test_idx]
        
        print(f"Data loaded. Train: {len(self.x_train)}, Test: {len(self.x_test)}")
        print(f"Input shape: {self.x_train.shape[1:]}")
        print(f"Target Min: {self.t_train.min(axis=0)}")
        print(f"Target Max: {self.t_train.max(axis=0)}")
        print(f"Target Mean: {self.t_train.mean(axis=0)}")

    # def _init_network(self):
    #     """CNNの構築"""
    #     input_dim = self.x_train.shape[1:] # (C, H, W)
        
    #     print("Initializing Network...")
    #     self.network = DeepConvNet(
    #         input_dim=input_dim,
    #         hidden_size=100,
    #         output_size=2,   # Steering, Throttle
    #         weight_init_std=0.01
    #     )

    def _init_network(self):
            """CNNの構築"""
            input_dim = self.x_train.shape[1:] 
            
            print("Initializing DeepConvNet...")
            
            # DeepConvNet の初期化
            # 3x3 の小さなフィルターを重ねるのが最近の主流（VGGスタイル）です
            self.network = DeepConvNet(
                input_dim=input_dim,
                
                # 1層目: フィルター16枚, サイズ3x3
                conv_param_1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                
                # 2層目: フィルター32枚, サイズ3x3
                conv_param_2={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                
                hidden_size=100,
                output_size=2
            )
            
            # オプティマイザは Adam (学習率は低めに設定)
            self.optimizer = Adam(lr=0.0001)

    def train(self, iters_num=1000):
        """学習ループの実行"""
        print(f"\nStart Training for {iters_num} iterations...")
        train_size = self.x_train.shape[0]
        
        for i in trange(iters_num):
            # ミニバッチ取得
            batch_mask = np.random.choice(train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            # 勾配計算
            grad = self.network.gradient(x_batch, t_batch)
            
            self.optimizer.update(self.network.params, grads=grad)
            # # パラメータ更新
            # for key in self.network.params.keys():
            #     self.network.params[key] -= self.learning_rate * grad[key]
            
            # 進捗表示
            if i % 10 == 0:
                loss = self.network.loss(x_batch, t_batch)
                print(f"Iter {i:4d} | Loss: {loss:.5f}")

        print("Training finished.")
        self.evaluate()

    def evaluate(self):
        """テストデータでの評価"""
        loss = self.network.loss(self.x_test, self.t_test)
        print(f"\nFinal Test Loss (MSE): {loss:.5f}")
        
        # 数件の予測結果を表示して直感的な確認を行う
        print("--- Prediction Sample ---")
        indices = np.random.choice(len(self.x_test), min(3, len(self.x_test)), replace=False)
        for idx in indices:
            img = self.x_test[idx].reshape(1, *self.x_test.shape[1:])
            pred = self.network.predict(img)[0]
            true = self.t_test[idx]
            print(f"True: [Str:{true[0]:.2f}, Thr:{true[1]:.2f}] -> Pred: [Str:{pred[0]:.2f}, Thr:{pred[1]:.2f}]")

    def save_weights(self):
        """重みをバイナリ形式で保存"""
        print(f"\nSaving weights to {self.output_dir}...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        for key, param in self.network.params.items():
            file_path = os.path.join(self.output_dir, f"{key}.bin")
            param.astype('float32').tofile(file_path)
            print(f"Saved {key} ({param.shape})")

if __name__ == "__main__":
    # 使用例
    try:
        # ディレクトリは環境に合わせて修正してください
        trainer = Trainer(data_dir="./data", batch_size=32, learning_rate=0.0001)
        
        # 学習実行
        trainer.train(iters_num=1000)
        
        # 重み保存
        trainer.save_weights()
        
    except Exception as e:
        print(f"An error occurred: {e}")
