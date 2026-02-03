import numpy as np
import logging
from abc import ABC, abstractmethod
from cnn.util import im2col, col2im
from cnn.parameter import Parameter
from functools import wraps

logger = logging.getLogger(__name__)


def log_shapes(func):
    """
    forward / backwardの入出力のSHAPEを出力するデコレーター
    use as `LOG_LEVEL=DEBUG make`
    """
    @wraps(func)
    def _wrapper(self, x: np.ndarray, *args, **kwargs):
        method_name = func.__name__
        result = func(self, x, *args, **kwargs)

        if hasattr(self, 'W') and hasattr(self, 'b'):
            log_msg = f"[{self.__class__.__name__}] {method_name} IN[{x.shape}]:W[{self.W.data.shape}]:b[{self.b.data.shape}]:OUT[{result.shape}]"
        else:
            log_msg = f"[{self.__class__.__name__}] {method_name} IN[{x.shape}] | OUT[{result.shape}]"
        logger.debug(log_msg)
        return result
    return _wrapper


class Layer(ABC):
    def __init__(self) -> None:
        self.training = True

    def train(self) -> None:
        self.training = True
        
    def eval(self) -> None:
        self.training = False

    @abstractmethod
    def forward(self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout:np.ndarray) -> np.ndarray:
        pass

    def parameters(self) -> list[Parameter]:
        params = []
        for name, value in vars(self).items(): #vars()で__dict__が取れるらしい？！
            if isinstance(value, Parameter):
                if value.name is None:
                    value.name = name
                params.append(value)
        return params


class Affine(Layer):
    """全結合層"""
    def __init__(self, input_size: int, output_size: int, name: str="affine") -> None:
        # 親クラスinitにてトレイニングかどうかのフラグを持つので、それを初期化。トレイニングならDROPOUT
        super().__init__()
        # Heの初期値 p183 Reluの場合これがいいらしい
        scale: float = np.sqrt(2.0 / input_size)
        # 正規分布にしたがった行列（input size, output size)
        W_data: np.ndarray = scale * np.random.randn(input_size, output_size)
        # ouput-sizeの行列
        b_data: np.ndarray = np.zeros(output_size)

        self.W = Parameter(W_data, name=f"{name}:W")
        self.b = Parameter(b_data, name=f"{name}:b")

        # 順伝播時に初期化、逆伝播のために保持
        self.x = None

    @log_shapes
    def forward(self, x:np.ndarray) -> np.ndarray:
        self.x = x
        out = np.dot(x, self.W.data) + self.b.data
        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.x is not None, "x must not be None"

        # xへの逆伝播を継続
        dx = np.dot(dout, self.W.data.T)

        # Weightsの勾配をParamクラスに保存。最終的にはoptimizerによって、勾配＊学習レシオ分だけウェイトを変更
        self.W.grad = np.dot(self.x.T, dout)
        # バイアルの勾配
        self.b.grad = np.sum(dout, axis=0)

        return dx


class Relu(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x is not None, "x cannot be None"
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    @log_shapes
    def backward(self, dout:np.ndarray) -> np.ndarray:
        assert self.mask is not None, "Run forward before backward"
        assert dout.shape == self.mask.shape, f"Shape mismatch: dout {dout.shape} != mask {self.mask.shape}"
        dout[self.mask] = 0
        dx = dout
        return dx


class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int=1, pad: int=0) -> None:
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.arg_max is not None, "Pooling arg_max is None"

        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class Convolution(Layer):
    def __init__(self, in_channels: int, out_channels: int, filter_size: int, stride: int=1, pad: int=0, name: str="Conv"):
        super().__init__()
        self.stride = stride
        self.pad = pad
        input_nodes = in_channels * filter_size * filter_size # 入力CH＊フィルター高さ＊フィルター幅
        scale = np.sqrt(2.0 / input_nodes)

        W_data = scale * np.random.randn(out_channels, in_channels, filter_size, filter_size)
        b_data = np.zeros(out_channels)

        self.W = Parameter(W_data, name=f"{name} W:")
        self.b = Parameter(b_data, name=f"{name} b:")

        self.x = None
        self.col = None
        self.col_W = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.data.shape
        N, C, H, W = x.shape
        out_h = 1 + (H + 2*self.pad - FH) // self.stride
        out_w = 1 + (W + 2*self.pad - FW) // self.stride

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.data.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b.data
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.data.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.b.grad = np.sum(dout, axis=0)

        dW = np.dot(self.col.T, dout)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
        self.W.grad = dW

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        # 入力の形状を記憶（(N, C, H, W)など）
        self.input_shape = x.shape
        # 1次元目（バッチサイズ）以外を平坦化
        out = x.reshape(x.shape[0], -1)
        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # 記憶しておいた形状に戻す
        dx = dout.reshape(*self.input_shape)
        return dx


class MeanSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None # AIの予測値
        self.t = None # 正解データ

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = y

        # 誤差の計算: 0.5 * (予測 - 正解)^2 の平均
        # 0.5を掛けるのは、微分の計算を綺麗にするため（慣習）
        batch_size = self.y.shape[0]
        self.loss = 0.5 * np.sum((self.y - self.t)**2) / batch_size

        return self.loss

    def backward(self, dout: float=1.0) -> np.ndarray:
        batch_size = self.t.shape[0]

        # 勾配: (予測 - 正解) / バッチサイズ
        dx = (self.y - self.t) / batch_size

        return dx

class Dropout(Layer):
    def __init__(self, dropout_ratio: float=0.5) -> None:
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask
