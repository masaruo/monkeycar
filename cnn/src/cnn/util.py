import numpy as np
import logging


logger = logging.getLogger(__name__)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    0-255の画像を0.0-1.0
    """
    return image.astype(np.float32) / 255.0


def im2col(
    input_data: np.ndarray, filter_h: int, filter_w: int, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    """
    基本的にpad = 0, stride = 1で考えていい
    その場合、out_h / out_x はスライドする回数。
    H = 10, filter_h = 2, out_h = 9。一個ずつ下にずれなていく
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # pad = 0なら無視していい
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    # ６次元の配列を作成
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w): #!予選データ収集後に修正。影響未知数
            x_max = x + stride * out_w
            """
            画像全体のうち特定の相対位置ピクセルを一斉に抜き出す。
            numply sliceは start:stop:step
            """
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    """
    * transpose
    [N, C, filter_h, filter_w, out_h, out_w] -> [N, out_h, out_w, C, filter_h, filter_w]
    [N, out_h(出力高さ), out_w（出力幅）] 一回の畳み込み計算の地点
    [C, filter_h（フィルタ高さ）, filter_w（フィルタ幅）] その地点の計算対象全データ

    * reshape
    一次元：N * out_h * out_w = 全バッチの全スライド位置を行列の行
    ２次元(-1 = 残り全部)： C * filter_h * filter_wを行列の列
    """
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w :
    stride :
    pad :

    Returns
    -------
    img :
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(
        0, 3, 4, 5, 1, 2
    )

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    out = img[:, :, pad : H + pad, pad : W + pad]
    return out
