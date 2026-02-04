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

        if hasattr(self, "W") and hasattr(self, "b"):
            log_msg = f"[{self.__class__.__name__}] {method_name} IN[{x.shape}]:W[{self.W.data.shape}]:b[{self.b.data.shape}]:OUT[{result.shape}]"
        else:
            log_msg = f"[{self.__class__.__name__}] {method_name} IN[{x.shape}] | OUT[{result.shape}]"
        # logger.debug(log_msg)
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
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass

    def parameters(self) -> list[Parameter]:
        params = []
        for name, value in vars(self).items():  # vars()で__dict__が取れるらしい？！
            if isinstance(value, Parameter):
                if value.name is None:
                    value.name = name
                params.append(value)
        return params


class Affine(Layer):
    """全結合層"""
    def __init__(self, input_size: int, output_size: int, name: str = "affine") -> None:
        # 親クラスinitにてTrainingかどうかのフラグを持つので、それを初期化。TrainingならDROPOUT Active
        super().__init__()
        # Heの初期値 p183 Reluの場合これがいいらしい
        scale: float = np.sqrt(2.0 / input_size)
        # 正規分布にしたがった行列（input size, output size)
        W_data: np.ndarray = scale * np.random.randn(input_size, output_size)
        # ouput-sizeの行列
        b_data: np.ndarray = np.zeros(output_size)

        self.W = Parameter(W_data, name=f"{name}:W") #(input_size, output_size)
        self.b = Parameter(b_data, name=f"{name}:b") #(output_size, )

        # 順伝播時に初期化、逆伝播のために保持
        self.x = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        """全結合層の順伝播

        Args:
            x (np.ndarray): (batch_size N、input_size)

        Returns:
            np.ndarray: _description_
        """
        self.x = x # 逆伝播の計算のため保持
        out = np.dot(x, self.W.data) + self.b.data
        logger.debug(f"Affine_forward: x{x.shape} @ W{self.W.data.shape} + b{self.b.data.shape} -> {out.shape}")

        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        逆伝播は責任のなすりつけ。ロスが３つに分解
        dout:この層の出力は、こうしてほしいというメッセージ
        dW（重み）：重みが悪い。dWを修正
        db（バイアスの責任）：バイアスが悪い。dbを修正
        dx（入力の責任）：そもそも、前から送られてきたデータXが悪い。送り返す
        """
        assert self.x is not None, "x must not be None"

        # xへの逆伝播を継続
        dx = np.dot(dout, self.W.data.T)

        # Weightsの勾配をParamクラスに保存。Optimizerによって、`learningRate * grad`分だけウェイトを変更していく
        # z=xy, dz/dx = y, dz/dy = x :　逆にひっくり返る
        # x shape (N, 2), dout shape (N, 3) x.Transpose shape (2, N)で行列の積ができる
        self.W.grad = np.dot(self.x.T, dout)
        # バイアスの勾配
        self.b.grad = np.sum(dout, axis=0)

        logger.debug(f"Affine_backward: dout{dout.shape} @ W.T{self.W.data.T.shape} -> dx{dx.shape}")
        return dx


class Relu(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.mask = None

    @log_shapes
    def forward(self, x: np.ndarray) -> np.ndarray:
        assert x is not None, "x cannot be None"
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "Run forward before backward"
        assert dout.shape == self.mask.shape, (
            f"Shape mismatch: dout {dout.shape} != mask {self.mask.shape}"
        )
        dout[self.mask] = 0
        dx = dout
        return dx


class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, pad: int = 0) -> None:
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

        logger.debug(f"Pooling_forward: In({N},{C},{H},{W}) P:{self.pad} S:{self.stride} -> im2col")
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        logger.debug(f"  im2col -> col{col.shape}")
        col = col.reshape(-1, self.pool_h * self.pool_w)
        logger.debug(f"  reshape -> col{col.shape}")

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        logger.debug(f"  max pooling -> out{out.shape}")
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        logger.debug(f"  reshape & transpose -> Out({N},{C},{out_h},{out_w})")

        self.x = x
        self.arg_max = arg_max

        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self.arg_max is not None, "Pooling arg_max is None"

        logger.debug(f"Pooling_backward: In{dout.shape}")
        dout = dout.transpose(0, 2, 3, 1)
        logger.debug(f"  transpose(0,2,3,1) -> {dout.shape}")

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        logger.debug(f"  dmax zeros -> {dmax.shape}")

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        logger.debug(f"  scatter gradient -> {dmax.shape}")

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        logger.debug(f"  reshape -> dcol{dcol.shape}")
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        logger.debug(f"  col2im -> dx{dx.shape}")

        return dx


class Convolution(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int = 1,
        pad: int = 0,
        name: str = "Conv",
    ):
        super().__init__()
        self.stride = stride
        self.pad = pad
        input_nodes = (
            in_channels * filter_size * filter_size
        )  # 入力CH＊フィルター高さ＊フィルター幅
        scale = np.sqrt(2.0 / input_nodes)

        W_data = scale * np.random.randn(
            out_channels, in_channels, filter_size, filter_size
        )
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
        out_h = 1 + (H + 2 * self.pad - FH) // self.stride
        out_w = 1 + (W + 2 * self.pad - FW) // self.stride

        logger.debug(f"Conv_forward: In({N},{C},{H},{W}), Filter({FN},{C},{FH},{FW}), Pad:{self.pad} Stride:{self.stride}")
        logger.debug(f"  -> Out({N},{FN},{out_h},{out_w})")

        col = im2col(x, FH, FW, self.stride, self.pad)
        logger.debug(f"  im2col -> col{col.shape}")
        col_W = self.W.data.reshape(FN, -1).T
        logger.debug(f"  reshape filter -> col_W{col_W.shape}")

        out = np.dot(col, col_W) + self.b.data
        logger.debug(f"  col @ col_W + b -> {col.shape} @ {col_W.shape} + {self.b.data.shape} = {out.shape}")
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        logger.debug(f"  reshape & transpose -> Out({N},{FN},{out_h},{out_w})")

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    @log_shapes
    def backward(self, dout: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.data.shape
        logger.debug(f"Conv_backward: In{dout.shape}")
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        logger.debug(f"  transpose(0,2,3,1) & reshape -> dout{dout.shape}")

        self.b.grad = np.sum(dout, axis=0)
        logger.debug(f"  sum for bias grad -> b.grad{self.b.grad.shape}")

        dW = np.dot(self.col.T, dout)
        logger.debug(f"  col.T @ dout -> {self.col.T.shape} @ {dout.shape} = {dW.shape}")
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
        logger.debug(f"  reshape -> W.grad({FN},{C},{FH},{FW})")
        self.W.grad = dW

        dcol = np.dot(dout, self.col_W.T)
        logger.debug(f"  dout @ col_W.T -> {dout.shape} @ {self.col_W.T.shape} = {dcol.shape}")
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        logger.debug(f"  col2im -> dx{dx.shape}")

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
        self.y = None  # 予測値
        self.t = None  # 正解データ

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = y

        # 誤差の計算: 0.5 * (予測 - 正解)^2 の平均
        # 0.5を掛けるのは、微分の計算を綺麗にするため。^2 -> 2 * 0.5 = 1
        batch_size = self.y.shape[0]
        self.loss = 0.5 * np.sum((self.y - self.t) ** 2) / batch_size
        # self.loss = np.sum((self.y - self.t) ** 2) / batch_size
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        assert self.y is not None, "MSE: self.y is None"
        assert self.t is not None, "MSE: self.t is None"

        batch_size = self.t.shape[0]

        # 勾配: (予測 - 正解) / バッチサイズ
        # dL / dy = (y - t) / batch_size
        dx = (self.y - self.t) / batch_size
        # dx = (self.y - self.t) * 2 / batch_size

        return dx


class Dropout(Layer):
    def __init__(self, dropout_ratio: float = 0.5) -> None:
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
