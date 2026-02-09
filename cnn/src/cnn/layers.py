import numpy as np
import logging
from abc import ABC, abstractmethod
from cnn.util import im2col, col2im
from cnn.parameter import Parameter
from functools import wraps

logger = logging.getLogger(__name__)


class Layer(ABC):
    """ニューラルネットワークレイヤーの抽象基底クラス

    すべてのレイヤー（Affine、ReLU、Pooling、Convolutionなど）の基となるクラス。
    訓練・評価モードの管理とパラメータ管理機能を提供します。
    """
    def __init__(self) -> None:
        """レイヤーの初期化。訓練モードで開始します。"""
        self.training = True

    def train(self) -> None:
        """訓練モードに設定します（Dropoutなどが有効になります）"""
        self.training = True

    def eval(self) -> None:
        """評価モードに設定します（Dropoutなどが無効になります）"""
        self.training = False

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播（フォーワードパス）の抽象メソッド

        Args:
            x (np.ndarray): 入力データ

        Returns:
            np.ndarray: 出力データ
        """
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """逆伝播（バックワードパス）の抽象メソッド

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配
        """
        pass

    def parameters(self) -> list[Parameter]:
        """このレイヤーが保有するパラメータ（重み、バイアスなど）を取得します

        Returns:
            list[Parameter]: Parameterオブジェクトのリスト
        """
        params = []
        for name, value in vars(self).items():  # vars()で__dict__が取れるらしい？！
            if isinstance(value, Parameter):
                if value.name is None:
                    value.name = name
                params.append(value)
        return params


class Affine(Layer):
    """全結合層（Fully Connected Layer）

    入力を全出力ユニットに接続し、重み付き線形変換を行う層。
    出力 = 入力 @ W + b
    """
    def __init__(self, input_size: int, output_size: int, name: str = "affine") -> None:
        """全結合層の初期化

        Args:
            input_size (int): 入力次元数
            output_size (int): 出力次元数
            name (str, optional): レイヤー名。デフォルトは "affine"
        """
        # 親クラスinitにてTrainingかどうかのフラグを持つので、それを初期化。TrainingならDROPOUT Active
        super().__init__()

        # Heの初期値 p183 Reluの場合これがいいらしい
        scale: float = np.sqrt(2.0 / input_size)

        # Wの行列を初期化（input_size, output_size)
        W_data: np.ndarray = scale * np.random.randn(input_size, output_size)

        # バイアスはoutput_sizeの1次元ベクトルで初期化(output_size, )
        b_data: np.ndarray = np.zeros(output_size)

        self.W = Parameter(W_data, name=f"{name}:W") #(input_size, output_size)
        self.b = Parameter(b_data, name=f"{name}:b") #(output_size, )

        # 順伝播時に初期化、逆伝播のために保持
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """全結合層の順伝播

        Args:
            x (np.ndarray): (batch_size、input_size)
        Returns:
            np.ndarray: xW(重みつき和) + b（バイアス）

        x(batch_size, input_size) dot W(input_size, output_size) -> (batch_size, output_size)の行列を順伝播へ送り出し
        """
        self.x = x
        # バイアスはNUMPYによりブロードキャストされてすべてのバッチに影響を与えている
        out = np.dot(x, self.W.data) + self.b.data
        logger.debug(f"Affine_FWD: x{x.shape} dot W{self.W.data.shape} + b{self.b.data.shape} -> {out.shape}")
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        逆伝播は責任のなすりつけ。誤差の原因を３つに分解
        dout:この層の出力は、こうしてほしいというメッセージ。勾配
        dW（重み）：重みが悪い。dWを修正
        db（バイアスの責任）：バイアスが悪い。dbを修正
        dx（入力の責任）：そもそも、前から送られてきたデータXが悪い。前方に逆伝播で差し戻す
        """
        assert self.x is not None, "x must not be None"

        """
        Wの計算:
        順伝播: y = x dot W + b
        逆伝播 dL / dw = (dL / dy) dot (dy / dw)
        (dL / dy) = dout
        (dy / dw) = x
        サイズが合わないので転置
        あとで、オプティマイザー経由で勾配をもとに変更
        """
        self.W.grad = np.dot(self.x.T, dout)

        """
        bの計算：
        順伝播：y = x dot W + b
        逆伝播: dL/db = (dL/dy) dot (dy/db)
        (dL/dy) = dout
        (dy/db) = 1、つまりdoutのみ

        ただし！バッチを複数にすると、ひとつひとつ影響をあたえるから、全部の影響の責任をとらなきゃならない

        (dL/dy) = SUM(dL / dy) dot (dy/db)
        (dy/db) = 1,
        (dL/dy) = SUM(dL / dy)
        あとで、オプティマイザー経由で勾配をもとに変更 
        """
        self.b.grad = np.sum(dout, axis=0)

        """
        dxの計算：
        順伝播：y = x dot W + b
        逆伝播：dL / dx = (dL / dy) dot (dy / dx)
        dL / dy = dout
        dy / dx = W
        dL / dx = dout dot W
        サイズ合わせに転置
        逆伝播を継続のために、リターン
        """
        dx = np.dot(dout, self.W.data.T)
        logger.debug(f"Affine_BACK: dout{dout.shape} dot W.T{self.W.data.T.shape} -> dx{dx.shape}")
        return dx


class Relu(Layer):
    """ReLU（Rectified Linear Unit）活性化関数層

    入力がx <= 0の場合は0を出力し、x > 0の場合はxをそのまま出力します。
    出力 = max(0, x)
    """
    def __init__(self) -> None:
        """ReLU層の初期化"""
        super().__init__()
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """ReLU活性化の順伝播

        Args:
            x (np.ndarray): 入力データ

        Returns:
            np.ndarray: max(0, x)の結果
        """
        assert x is not None, "x cannot be None"
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """ReLUの逆伝播

        入力がx <= 0の場所では勾配を0とし、x > 0の場所では勾配をそのまま通します。

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配
        """
        assert self.mask is not None, "Run forward before backward"
        assert dout.shape == self.mask.shape, (
            f"Shape mismatch: dout {dout.shape} != mask {self.mask.shape}"
        )
        dout[self.mask] = 0
        dx = dout
        return dx


class Pooling(Layer):
    """Max Pooling層

    入力の局所領域ごとに最大値を取り出す層。
    特徴マップのダウンサンプリングに使用され、計算量削減と空間的な不変性を実現。
    """
    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, pad: int = 0) -> None:
        """Pooling層の初期化

        Args:
            pool_h (int): プーリング領域の高さ
            pool_w (int): プーリング領域の幅
            stride (int, optional): ストライド（移動幅）。デフォルトは1
            pad (int, optional): パディング（0埋め）。デフォルトは0
        """
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Max Poolingの順伝播

        Args:
            x (np.ndarray): 入力テンソル。形状: (batch_size, channels, height, width)

        Returns:
            np.ndarray: プールされた出力テンソル
        """
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 全バッチの全スライド位置の行列の行とC, pool_h, pool_wの行列の列が帰ってきている
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # [N * OH * OW, C * PH * PW] => [N * OH * OW * C, PH * PW]
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # axis=1によってPH * PWの最大値の場所をGET
        arg_max = np.argmax(col, axis=1)
        # PW * PWの最大値をGET。全てのバッチ、チャネル、出力位置の最大値が一列の一次元配列
        out = np.max(col, axis=1)
        """
        * reshape
        ([N * OH * OW *C],[PH * PW])の二次元を
            => (N, out_h, out_w, C)の四次元に戻す？
        * transpose
        N, C, H, Wのフォーマットに変更
        """
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x

        # 逆伝播時、最大値を与えた要素のみに勾配を伝え、それ以外は０にする必要あり
        self.arg_max = arg_max

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Max Poolingの逆伝播

        最大値を取った位置に勾配を流し、その他の位置には勾配0

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配
        """
        assert self.arg_max is not None, "Pooling arg_max is None"

        # (N,C,H,W)->(N,H,W,C)
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # 全要素がゼロ (N*OH*OW*C, poll_size)
        dmax = np.zeros((dout.size, pool_size))

        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # (N*OH*OW, C*pool_h*pool_w)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # -> (N,C,H,W)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class Convolution(Layer):
    """Convolution層（畳み込み層）

    フィルターを入力画像上でスライドさせながら重み付き和を計算する層
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        stride: int = 1,
        pad: int = 0,
        name: str = "Conv",
    ):
        """Convolution層の初期化

        Args:
            in_channels (int): 入力チャネル数
            out_channels (int): 出力フィルター数（チャネル数）
            filter_size (int): フィルターサイズ（正方形を仮定）
            stride (int, optional): ストライド（移動幅）。デフォルトは1
            pad (int, optional): パディング（0埋め）。デフォルトは0
            name (str, optional): レイヤー名。デフォルトは "Conv"
        """
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
        self.col_x = None
        self.col_W = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Convolution層の順伝播

        Args:
            x (np.ndarray): 入力テンソル。形状: (batch_size, in_channels, height, width)

        Returns:
            np.ndarray: 出力テンソル。形状: (batch_size, out_channels, out_height, out_width)
        """
        FN, C, FH, FW = self.W.data.shape
        N, C, H, W = x.shape
        OH = 1 + (H + 2 * self.pad - FH) // self.stride
        OW = 1 + (W + 2 * self.pad - FW) // self.stride

        # (N, C, H, W) -> (N*OH*OW, C*FH*FW)
        col_x = im2col(x, FH, FW, self.stride, self.pad)
        # (FN, C, FH, FW) -> (FN, C*FH*FW) ->(C*FH*FW, FN)
        col_W = self.W.data.reshape(FN, -1).T

        # -> (N*OH*OW, FN)
        out = np.dot(col_x, col_W) + self.b.data
        # (N*OH*OW, FN) -> (N, OH, OW, FN) -> (N, FN, OH, OW)
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col_x = col_x
        self.col_W = col_W

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Convolution層の逆伝播

        重みとバイアスの勾配を計算し、入力側への誤差勾配を返します。

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配
        """
        FN, C, FH, FW = self.W.data.shape
        # (N,C,OH,OW)->(N,OH,OW,FN)->(N*OH*OW,FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # Check affine
        self.b.grad = np.sum(dout, axis=0)

        # (C*FH*FW, N*OH*OW) dot (N*OH*OW, FN) -> (C*FH*FW, FN)
        dW = np.dot(self.col_x.T, dout)
        # (C*FH*FW, FN) -> (FN, C*FH*FW) -> (FN, C, FH, FW)
        dW = dW.transpose(1, 0).reshape(FN, C, FH, FW)
        self.W.grad = dW

        # (N*C*OH*OW, FN) dot (FN, C*FH*FW)-> (N*C*OH*OW, C*FH*FW)
        dcol = np.dot(dout, self.col_W.T)
        # -> (N, C, H, W)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Flatten(Layer):
    """Flatten層（平坦化層）

    多次元テンソル（例：(batch_size, channels, height, width)）を
    2次元テンソル（batch_size, features）に変形する層。
    ConvolutionレイヤーとAffineレイヤーの間に使用。
    """
    def __init__(self):
        """Flatten層の初期化"""
        super().__init__()
        self.input_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Flatten層の順伝播

        Args:
            x (np.ndarray): 入力テンソル（任意の形状）

        Returns:
            np.ndarray: 2次元テンソル。形状: (batch_size, -1)
        """
        # 入力の形状を記憶（(N, C, H, W)など）
        self.input_shape = x.shape
        # 1次元目（バッチサイズ）以外を平坦化
        out = x.reshape(x.shape[0], -1)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Flatten層の逆伝播

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配（元の形状に復元）
        """
        # 記憶しておいた形状に戻す
        dx = dout.reshape(*self.input_shape)
        return dx


class MeanSquaredError:
    """平均二乗誤差（Mean Squared Error）損失関数

    回帰問題における損失関数。予測値と正解値の差の二乗の平均を計算します。
    """
    def __init__(self):
        """MeanSquaredError損失関数の初期化"""
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        """平均二乗誤差（Mean Squared Error）の計算

        バッチ内の各サンプルについて、予測値と正解値の差の二乗の平均を計算する。
        微分計算の簡略化のため、0.5 を掛ける。

        Args:
            y (np.ndarray): 予測値（モデルの出力）。形状: (batch_size, ...)
            t (np.ndarray): 正解データ（ターゲット）。形状: (batch_size, ...)

        Returns:
            float: MSE損失値
        """
        self.t = t
        self.y = y

        batch_size = self.y.shape[0]
        self.loss = 0.5 * np.sum((self.y - self.t) ** 2) / batch_size #微分すると２が降りてくるので、2 * 0.5で１になるので計算が簡単になる
        return self.loss

    def backward(self, dout: float = 1.0) -> np.ndarray:
        """平均二乗誤差の逆伝播（勾配計算）

        Args:
            dout (float, optional): 上流からの勾配。デフォルトは1.0

        Returns:
            np.ndarray: 予測値に対する損失関数の勾配
        """
        assert self.y is not None, "MSE: self.y is None"
        assert self.t is not None, "MSE: self.t is None"

        batch_size = self.t.shape[0]

        """
        dxの計算： yが推論値、tが教師データ
        順伝播: L = 1/N * SUM(Ei) 
                where E = 0.5 * (y - t)^2
        逆伝播: dL / dyで、ロスを減らすにはyの推論値をどれだけ変化させるべきかを計算
        dL / dy = (dL / dE) dot (dE / dy)
        dL / dE = 1 / N
        dE / dy = 0.5 * 2(y - t) => (y - t)
        dL / dy = 1 / N * (y - t)
        """
        # 勾配: (予測 - 正解) / バッチサイズ
        dx = (self.y - self.t) / batch_size

        return dx


class Dropout(Layer):
    """Dropout層（ドロップアウト層）

    訓練時にランダムにニューロンを無効化することで、過学習を防ぐ正則化技法。
    評価時はドロップアウトを適用せず、スケーリングのみ行います。
    """
    def __init__(self, dropout_ratio: float = 0.5) -> None:
        """Dropout層の初期化

        Args:
            dropout_ratio (float, optional): ドロップアウト確率（0-1）。デフォルトは0.5
        """
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Dropout層の順伝播

        訓練モード時はランダムにニューロンを無効化し、評価モード時はスケーリングを適用します。

        Args:
            x (np.ndarray): 入力データ

        Returns:
            np.ndarray: ドロップアウト適用後のデータ
        """
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Dropout層の逆伝播

        順伝播で適用したマスクと同じマスクを勾配に適用します。

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: マスク適用後の誤差勾配
        """
        return dout * self.mask


class Tanh(Layer):
    """Tanh（双曲線正接）活性化関数層

    入力を-1から1の範囲に圧縮する活性化関数。
    ReLUと比較して、負の値も処理でき、勾配消失が起きやすいという特性があるらしいが、使用せず
    """
    def __init__(self):
        """Tanh層の初期化"""
        super().__init__()
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Tanh活性化の順伝播

        Args:
            x (np.ndarray): 入力データ

        Returns:
            np.ndarray: tanh(x)の結果（-1 ~ 1の範囲）
        """
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Tanhの逆伝播

        tanh(x)の微分は(1 - tanh^2(x))です。

        Args:
            dout (np.ndarray): 出力側からの誤差勾配

        Returns:
            np.ndarray: 入力側への誤差勾配
        """
        dx = dout * (1.0 - self.out ** 2)
        return dx
