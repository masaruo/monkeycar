import logging
import numpy as np
import logging
from shared.layer import Convolution, Relu, Affine, MeanSquaredErrorWithLoss, Pooling
from collections import OrderedDict


logger = logging.getLogger(__name__)


class ConvNetwork:
    def __init__(self, input_dim=(3, 120, 160), output_size=2) -> None:
        # 1. パラメータの初期化 (W1, b1, W2, b2... を生成)
        self.params = self.init_params(input_dim, output_size)
        self.layers = OrderedDict()

        # 2. 畳み込みブロックの構築 (パラメータをレイヤに注入)
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], stride=1, pad=0)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], stride=1, pad=0)
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], stride=1, pad=0)
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)

        # 3. 全結合ブロックの構築
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu() # Affineの間にも活性化関数を挟むのが標準的

        self.layers['Affine2'] = Affine(self.params['W5'], self.params['b5'])

        # 4. 最終層 (損失関数)
        self.last_layer = MeanSquaredErrorWithLoss()

    def init_params(self, input_dim=(3, 120, 160), output_size=2):
        """
        input_dim: (Channels, Height, Width)
        output_size: ステアリングとスロットルの2
        """
        params = {}
        
        # --- 1. 畳み込み層の設計パラメータ ---
        # {レイヤ名: (フィルタ数, フィルタサイズ, ストライド, パディング)}
        # Donkey Carの標準に近い構成を採用
        conf = {
            'C1': {'n': 24, 'f': 5},
            'C2': {'n': 32, 'f': 5},
            'C3': {'n': 64, 'f': 3}
        }

        # --- 2. 重みの初期化 (Heの初期値) ---
        # Layer 1: Conv (in_ch=3, out_ch=24)
        n1 = input_dim[0] * conf['C1']['f']**2
        params['W1'] = np.random.randn(conf['C1']['n'], input_dim[0], conf['C1']['f'], conf['C1']['f']) * np.sqrt(2 / n1)
        params['b1'] = np.zeros(conf['C1']['n'])

        # Layer 2: Conv (in_ch=24, out_ch=32)
        n2 = conf['C1']['n'] * conf['C2']['f']**2
        params['W2'] = np.random.randn(conf['C2']['n'], conf['C1']['n'], conf['C2']['f'], conf['C2']['f']) * np.sqrt(2 / n2)
        params['b2'] = np.zeros(conf['C2']['n'])

        # Layer 3: Conv (in_ch=32, out_ch=64)
        n3 = conf['C2']['n'] * conf['C3']['f']**2
        params['W3'] = np.random.randn(conf['C3']['n'], conf['C2']['n'], conf['C3']['f'], conf['C3']['f']) * np.sqrt(2 / n3)
        params['b3'] = np.zeros(conf['C3']['n'])

        # --- 3. ダミーデータによる Flatten サイズの自動計算 ---
        # ここで一度 Convolution + Pooling 層を組んで、出口のサイズを取得する
        flatten_size = self._calculate_flatten_size(input_dim, params, conf)

        # --- 4. 全結合層（Affine）の初期化 ---
        # Layer 4: Affine1 (flatten_size -> 100)
        params['W4'] = np.random.randn(flatten_size, 100) * np.sqrt(2 / flatten_size)
        params['b4'] = np.zeros(100)

        # Layer 5: Affine2 (100 -> output_size)
        params['W5'] = np.random.randn(100, output_size) * np.sqrt(2 / 100)
        params['b5'] = np.zeros(output_size)

        return params

    def _calculate_flatten_size(self, input_dim, params, conf):
        
        # ダミーデータを生成 (1枚分)
        x = np.zeros((1, *input_dim))
        
        # 畳み込み層とプーリング層を順次適用
        x = Convolution(params['W1'], params['b1'], stride=1).forward(x)
        x = Pooling(pool_h=2, pool_w=2, stride=2).forward(x)
        
        x = Convolution(params['W2'], params['b2'], stride=1).forward(x)
        x = Pooling(pool_h=2, pool_w=2, stride=2).forward(x)
        
        x = Convolution(params['W3'], params['b3'], stride=1).forward(x)
        x = Pooling(pool_h=2, pool_w=2, stride=2).forward(x)
    
        return x.size # 全要素数を返す

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
        """
        x: 入力画像 (Batch, 3, 120, 160)
        t: 教師データ (Batch, 2) [steering, throttle]
        """
        # 1. Forward (損失を計算することで各層に中間データを保持させる)
        self.loss(x, t)

        # 2. Backward (最終層から逆向きに開始)
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 各レイヤに計算された勾配を収集する
        # self.layers['Conv1'] 等のインスタンスから dW, db を取り出す
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        
        grads['W4'], grads['b4'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W5'], grads['b5'] = self.layers['Affine2'].dW, self.layers['Affine2'].db


        return grads

    def loss(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        import pickle
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val

        # レイヤのパラメータも更新
        self.layers['Conv1'].W = self.params['W1']
        self.layers['Conv1'].b = self.params['b1']
        self.layers['Conv2'].W = self.params['W2']
        self.layers['Conv2'].b = self.params['b2']
        self.layers['Conv3'].W = self.params['W3']
        self.layers['Conv3'].b = self.params['b3']
        self.layers['Affine1'].W = self.params['W4']
        self.layers['Affine1'].b = self.params['b4']
        self.layers['Affine2'].W = self.params['W5']
        self.layers['Affine2'].b = self.params['b5']

    def evaluate(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        loss = self.last_layer.forward(y, t)
        return loss
