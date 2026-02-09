# This file contains code derived from "Deep Learning from Scratch"
# Copyright (c) 2016 Koki Saitoh
# Released under the MIT license
# https://github.com/oreilly-japan/deep-learning-from-scratch

import numpy as np
import logging


logger = logging.getLogger(__name__)


def softmax(x):
    """
    ソフトマックス関数 (オーバーフロー対策済み)
    x: 入力 (N, Class数) または (Class数,)
    """
    # バッチ処理対応: 入力が2次元の場合、各行ごとに計算
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)  # オーバーフロー対策
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    # 入力が1次元の場合
    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    # 1次元（データ1つ）できたら、2次元（バッチサイズ1）に変換して統一的に扱う
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    # log(0)を防ぐ微小値
    delta = 1e-7

    # 教師データtがOne-hot表現の場合、正解インデックスに変換して計算
    return -np.sum(t * np.log(y + delta)) / batch_size
