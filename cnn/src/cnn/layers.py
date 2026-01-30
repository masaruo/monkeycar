import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Self
from .functions import softmax, cross_entropy_error
from .util import im2col, col2im

logger = logging.getLogger(__name__)


class Layer(ABC):
    @abstractmethod
    def forward(self: Self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self: Self, dout:np.ndarray) -> np.ndarray:
        pass


class LossLayer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, dout: float = 1.0) -> np.ndarray:
        pass


class Affine(Layer):
    def __init__(self: Self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
        assert W.ndim == 2
        assert b.ndim == 1
        assert W.shape[1] == b.shape[0]

    def forward(self: Self, x:np.ndarray) -> np.ndarray:
        assert x.ndim == 2, f"x must be 2D, but got {x.ndim}"
        assert x.shape[1] == self.W.shape[0], f"Shape mismatch with x: {x.shape} and W: {self.W.shape}"

        self.x = x
        out: np.ndarray = np.dot(x, self.W) + self.b
        return out

    def backward(self: Self, dout: np.ndarray) -> np.ndarray:
        assert self.x is not None, "x must not be None"
        assert dout.shape[1] == self.W.T.shape[0], f"Shape mismatch"
        assert self.x.T.shape[1] == dout.shape[0], f"Shape mismatch"

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class Relu(Layer):
    def __init__(self: Self) -> None:
        self.mask = None

    def forward(self: Self, x: np.ndarray) -> np.ndarray:
        assert x is not None, "x cannot be None"
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self: Self, dout:np.ndarray) -> np.ndarray:
        assert self.mask is not None, "Run forward before backward"
        assert dout.shape == self.mask.shape, f"Shape mismatch: dout {dout.shape} != mask {self.mask.shape}"
        dout[self.mask] = 0
        dx = dout
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ(One-hot)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # バッチサイズを取得
        batch_size = self.t.shape[0]
        
        # 【ここが本と同じ単純な引き算】
        # yとtの形状が同じ（One-hot同士）なら、これでOK
        dx = (self.y - self.t) / batch_size
        
        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
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

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + (H + 2*self.pad - FH) // self.stride
        out_w = 1 + (W + 2*self.pad - FW) // self.stride

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        # 入力の形状を記憶（(N, C, H, W)など）
        self.input_shape = x.shape
        # 1次元目（バッチサイズ）以外を平坦化
        out = x.reshape(x.shape[0], -1)
        return out

    def backward(self, dout):
        # 記憶しておいた形状に戻す
        dx = dout.reshape(*self.input_shape)
        return dx


class MeanSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None # AIの予測値
        self.t = None # 正解データ

    def forward(self, y, t):
        self.t = t
        self.y = y
        
        # 誤差の計算: 0.5 * (予測 - 正解)^2 の平均
        # 0.5を掛けるのは、微分の計算を綺麗にするため（慣習）
        batch_size = self.y.shape[0]
        self.loss = 0.5 * np.sum((self.y - self.t)**2) / batch_size
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        # 勾配: (予測 - 正解) / バッチサイズ
        dx = (self.y - self.t) / batch_size
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio: float=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag: bool=True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
