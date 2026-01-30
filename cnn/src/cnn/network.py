import numpy as np
from collections import OrderedDict
import logging
from .layers import Affine, Relu, SoftmaxWithLoss, Convolution, Pooling, Flatten, MeanSquaredError, Dropout
import pickle

logger = logging.getLogger(__name__)


class TwolayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads


class SimpleConvNet:
    """単純なConvNet
    conv - relu - pool - affine - relu - affine - softmax
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=1, weight_init_std=0.01,
                 regression=True):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        input_C, input_H, input_W = input_dim
        
        # サイズ計算
        conv_output_h = (input_H - filter_size + 2*filter_pad) // filter_stride + 1
        conv_output_w = (input_W - filter_size + 2*filter_pad) // filter_stride + 1
        pool_output_h = int(conv_output_h / 2)
        pool_output_w = int(conv_output_w / 2)
        pool_output_size = filter_num * pool_output_h * pool_output_w

        # 重みの初期化 (Heの初期化: ReLUに最適化)
        self.params = {}
        
        # Conv1: 入力ノード数 = チャンネル数 * フィルタ面積
        pre_node_nums_1 = input_C * filter_size * filter_size
        w1_scale = np.sqrt(2.0 / pre_node_nums_1) # Heの初期値
        self.params['W1'] = w1_scale * np.random.randn(filter_num, input_C, filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        
        # Affine1
        pre_node_nums_2 = pool_output_size
        w2_scale = np.sqrt(2.0 / pre_node_nums_2)
        self.params['W2'] = w2_scale * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        
        # Affine2
        pre_node_nums_3 = hidden_size
        w3_scale = np.sqrt(2.0 / pre_node_nums_3)
        self.params['W3'] = w3_scale * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Flatten'] = Flatten()
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        if regression:
            self.last_layer = MeanSquaredError()
        else:
            self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        # 1. forward
        self.loss(x, t)

        # 2. backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 3. 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
            for key, val in params.items():
                self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]

# src/cnn/network.py に追記

# class DeepConvNet:
#     """
#     Kerasモデルに近づけた多層CNN
#     構成: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Affine -> ReLU -> Affine -> ReLU -> Affine(Out)
#     """
#     def __init__(self, input_dim=(3, 120, 160),
#                  conv_param_1={'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
#                  conv_param_2={'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
#                  hidden_size=100, output_size=2, weight_init_std=0.01):
        
#         # --- レイヤ1の出力サイズ計算 ---
#         input_C, input_H, input_W = input_dim
#         # Conv1
#         conv1_h = (input_H - conv_param_1['filter_size'] + 2*conv_param_1['pad']) // conv_param_1['stride'] + 1
#         conv1_w = (input_W - conv_param_1['filter_size'] + 2*conv_param_1['pad']) // conv_param_1['stride'] + 1
#         # Pool1 (2x2)
#         pool1_h = int(conv1_h / 2)
#         pool1_w = int(conv1_w / 2)
        
#         # --- レイヤ2の出力サイズ計算 ---
#         # Conv2
#         conv2_h = (pool1_h - conv_param_2['filter_size'] + 2*conv_param_2['pad']) // conv_param_2['stride'] + 1
#         conv2_w = (pool1_w - conv_param_2['filter_size'] + 2*conv_param_2['pad']) // conv_param_2['stride'] + 1
#         # Pool2 (2x2)
#         pool2_h = int(conv2_h / 2)
#         pool2_w = int(conv2_w / 2)
        
#         # 全結合層への入力サイズ
#         pool2_output_size = conv_param_2['filter_num'] * pool2_h * pool2_w

#         # 重みの初期化 (He初期化)
#         self.params = {}
        
#         # Layer 1: Conv
#         pre1 = input_C * conv_param_1['filter_size']**2
#         self.params['W1'] = np.sqrt(2.0/pre1) * np.random.randn(conv_param_1['filter_num'], input_C, conv_param_1['filter_size'], conv_param_1['filter_size'])
#         self.params['b1'] = np.zeros(conv_param_1['filter_num'])
        
#         # Layer 2: Conv
#         pre2 = conv_param_1['filter_num'] * conv_param_2['filter_size']**2
#         self.params['W2'] = np.sqrt(2.0/pre2) * np.random.randn(conv_param_2['filter_num'], conv_param_1['filter_num'], conv_param_2['filter_size'], conv_param_2['filter_size'])
#         self.params['b2'] = np.zeros(conv_param_2['filter_num'])
        
#         # Layer 3: Affine (Hidden)
#         pre3 = pool2_output_size
#         self.params['W3'] = np.sqrt(2.0/pre3) * np.random.randn(pool2_output_size, hidden_size)
#         self.params['b3'] = np.zeros(hidden_size)
        
#         # Layer 4: Affine (Output)
#         pre4 = hidden_size
#         self.params['W4'] = np.sqrt(2.0/pre4) * np.random.randn(hidden_size, output_size)
#         self.params['b4'] = np.zeros(output_size)

#         # レイヤの生成
#         self.layers = OrderedDict()
        
#         # Block 1
#         self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
#         self.layers['Relu1'] = Relu()
#         self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
#         # Block 2
#         self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
#         self.layers['Relu2'] = Relu()
#         self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
#         self.layers['Flatten'] = Flatten()
        
#         # Fully Connected
#         self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
#         self.layers['Relu3']   = Relu()
#         self.layers['Dropout1'] = Dropout(0.5)
#         self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

#         self.last_layer = MeanSquaredError()

class DeepConvNet:
    """
    Kerasモデル (3ブロック構成) に合わせた修正版
    構成: Conv(16)->Pool -> Conv(32)->Pool -> Conv(64)->Pool -> Flatten -> Affine -> ...
    """
    def __init__(self, input_dim=(3, 120, 160),
                 conv_param_1={'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2={'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1}, # 追加
                 hidden_size=100, output_size=2, weight_init_std=0.01):
        
        # --- レイヤ1の出力サイズ計算 ---
        input_C, input_H, input_W = input_dim
        conv1_h = (input_H - conv_param_1['filter_size'] + 2*conv_param_1['pad']) // conv_param_1['stride'] + 1
        conv1_w = (input_W - conv_param_1['filter_size'] + 2*conv_param_1['pad']) // conv_param_1['stride'] + 1
        pool1_h = int(conv1_h / 2)
        pool1_w = int(conv1_w / 2)
        
        # --- レイヤ2の出力サイズ計算 ---
        conv2_h = (pool1_h - conv_param_2['filter_size'] + 2*conv_param_2['pad']) // conv_param_2['stride'] + 1
        conv2_w = (pool1_w - conv_param_2['filter_size'] + 2*conv_param_2['pad']) // conv_param_2['stride'] + 1
        pool2_h = int(conv2_h / 2)
        pool2_w = int(conv2_w / 2)

        # --- レイヤ3の出力サイズ計算 (新規追加) ---
        conv3_h = (pool2_h - conv_param_3['filter_size'] + 2*conv_param_3['pad']) // conv_param_3['stride'] + 1
        conv3_w = (pool2_w - conv_param_3['filter_size'] + 2*conv_param_3['pad']) // conv_param_3['stride'] + 1
        pool3_h = int(conv3_h / 2)
        pool3_w = int(conv3_w / 2)
        
        # 全結合層への入力サイズ (劇的に小さくなる)
        # 例: (3,120,160) -> Pool3出力 (64, 15, 20) -> 19,200 (まだ大きめだが半減)
        # さらにフィルタ数を調整するか、ストライドを使えばもっと減る
        pool3_output_size = conv_param_3['filter_num'] * pool3_h * pool3_w

        # 重みの初期化
        self.params = {}
        
        # Layer 1
        pre1 = input_C * conv_param_1['filter_size']**2
        self.params['W1'] = np.sqrt(2.0/pre1) * np.random.randn(conv_param_1['filter_num'], input_C, conv_param_1['filter_size'], conv_param_1['filter_size'])
        self.params['b1'] = np.zeros(conv_param_1['filter_num'])
        
        # Layer 2
        pre2 = conv_param_1['filter_num'] * conv_param_2['filter_size']**2
        self.params['W2'] = np.sqrt(2.0/pre2) * np.random.randn(conv_param_2['filter_num'], conv_param_1['filter_num'], conv_param_2['filter_size'], conv_param_2['filter_size'])
        self.params['b2'] = np.zeros(conv_param_2['filter_num'])
        
        # Layer 3 (新規追加)
        pre3 = conv_param_2['filter_num'] * conv_param_3['filter_size']**2
        self.params['W3'] = np.sqrt(2.0/pre3) * np.random.randn(conv_param_3['filter_num'], conv_param_2['filter_num'], conv_param_3['filter_size'], conv_param_3['filter_size'])
        self.params['b3'] = np.zeros(conv_param_3['filter_num'])

        # Layer 4: Affine (Hidden) - 入力がpool3_output_sizeになる
        pre4 = pool3_output_size
        self.params['W4'] = np.sqrt(2.0/pre4) * np.random.randn(pool3_output_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        
        # Layer 5: Affine (Output)
        pre5 = hidden_size
        self.params['W5'] = np.sqrt(2.0/pre5) * np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        
        # Block 1
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # Block 2
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], conv_param_2['stride'], conv_param_2['pad'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # Block 3 (新規追加)
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'], conv_param_3['stride'], conv_param_3['pad'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        self.layers['Flatten'] = Flatten()
        
        # Fully Connected
        self.layers['Affine1'] = Affine(self.params['W4'], self.params['b4']) # W3->W4に変更
        self.layers['Relu4']   = Relu()
        self.layers['Dropout1'] = Dropout(0.5)
        self.layers['Affine2'] = Affine(self.params['W5'], self.params['b5']) # W4->W5に変更

        self.last_layer = MeanSquaredError()

    def predict(self, x, train_flag=False):
        for key, layer in self.layers.items():
            if "Dropout" in key:
                x = layer.forward(x, train_flag=True)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flag=True):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # def gradient(self, x, t):
    #     # forward
    #     self.loss(x, t, train_flag=True)

    #     # backward
    #     dout = 1
    #     dout = self.last_layer.backward(dout)

    #     layers = list(self.layers.values())
    #     layers.reverse()
    #     for layer in layers:
    #         dout = layer.backward(dout)

    #     grads = {}
    #     grads['W1'] = self.layers['Conv1'].dW
    #     grads['b1'] = self.layers['Conv1'].db
    #     grads['W2'] = self.layers['Conv2'].dW
    #     grads['b2'] = self.layers['Conv2'].db
    #     grads['W3'] = self.layers['Affine1'].dW
    #     grads['b3'] = self.layers['Affine1'].db
    #     grads['W4'] = self.layers['Affine2'].dW
    #     grads['b4'] = self.layers['Affine2'].db

    #     return grads

    def gradient(self, x, t):
            """
            誤差逆伝播法による勾配の算出
            Conv3を追加した構成に合わせて、gradsのキー割り当てを修正
            """
            # forward
            self.loss(x, t)

            # backward
            dout = 1
            dout = self.last_layer.backward(dout)

            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)

            # 設定の集約
            grads = {}
            # Block 1
            grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
            # Block 2
            grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
            # Block 3 (ここが抜けていたか、Affine1がW3に割り当てられていたのが原因)
            grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
            
            # Fully Connected (キーをW4, W5にシフト)
            grads['W4'], grads['b4'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W5'], grads['b5'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

            return grads
