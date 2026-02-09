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
    基本的にpad = 0, stride = 1で考えていいか。
    """
    N, C, H, W = input_data.shape

    # フィルタを何回スライドさせる必要があるのか？ H = 10, filter_h = 2の場合、９回
    OH = (H + 2 * pad - filter_h) // stride + 1
    OW = (W + 2 * pad - filter_w) // stride + 1

    # pad = 今回はゼロだから無視していい
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")

    # ６次元の配列を作成
    col = np.zeros((N, C, filter_h, filter_w, OH, OW))

    for y in range(filter_h):
        y_max = y + stride * OH
        for x in range(filter_w): #!予選データ収集後に修正。多分これまで、正正方のフィルターだったから問題なかったかな？
            x_max = x + stride * OW
            """
            画像全体のうち特定の相対位置ピクセルを一斉に抜き出す。
            numply sliceは start:stop:step
            colのy, xの時、(N, C, OH, OW) <= (:, :, y ~ y_maxの連続データ, x ~ x_maxの連続データ)
            y、xを指定すると、その場所は固定され、その次元は一時的に消える
            """
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    """
    transpose
    [N, C, filter_h, filter_w, OH, OW] -> [N, OH, OW, C, filter_h, filter_w]
    """
    transposed_col = col.transpose(0, 4, 5, 1, 2, 3)
    """
    reshape [N, OH, OW, C, filter_h, filter_w] -> ([N*OH*OW],[C*filter_h*filter_w])
    1dim: [N*OH*OW] フィルタを何回スライドさせなければいけないか？
    2dim: [C*filter_h（フィルタ高さ）*filter_w（フィルタ幅）] そのフィルタ時点での、フィルタに含まれる全チャネル・全画素を一列に並べたもの
    """
    reshaped_col = transposed_col.reshape(N * OH * OW, -1)
    return reshaped_col

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
