import numpy as np
import logging
from .layers import Affine, Relu, Convolution, Pooling, Flatten, MeanSquaredError, Dropout
import pickle
from typing import Union
from .parameter import Parameter

logger = logging.getLogger(__name__)


class CarConvNet:
    """
    3層の畳み込み層を持つCNN (Parameterクラス対応版)
    構成: Conv -> Relu -> Pool (x3) -> Flatten -> Affine -> Relu -> Dropout -> Affine -> MSE
    """
    def __init__(self, input_dim: tuple[int, int, int] = (3, 120, 160),
                conv_params: list[dict[str, int]] = [
                    {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':2},
                    {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                    {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 ],
                 hidden_size: int=100, output_size: int=2) -> None:

        self.layers = []
        c, h, w = input_dim

        # 畳み込み層
        for i, param in enumerate(conv_params):
            fn = param['filter_num']
            fs = param['filter_size']
            pad = param['pad']
            stride = param['stride']

            self.layers.append(Convolution(c, fn, fs, stride, pad))
            self.layers.append(Relu())
            self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

            h, w = self._get_convolution_outsize(h, w, fs, stride, pad)
            h, w = int(h/2), int(w/2)
            c = fn

        # 全結合層
        flat_size = c * h * w

        self.layers.append(Flatten())
        self.layers.append(Affine(flat_size, hidden_size))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(hidden_size, output_size))

        self.last_layer = MeanSquaredError()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y: np.ndarray = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x: np.ndarray, t: np.ndarray) -> None:
        # 1. Foward
        self.loss(x, t)

        #2. Backward
        dout: Union[float, np.ndarray] = 1.0
        dout = self.last_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return None

    def params(self) -> list[Parameter]:
        all_params: list[Parameter] = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                all_params.extend(layer.parameters())
        return all_params

    def train_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def save_params(self, file_name: str="params.pkl") -> None:
        params_dict: dict[str, np.ndarray] = {}

        for i, param in enumerate(self.params()):
            key: str = param.name if param.name else f"param_{i}"
            params_dict[key] = param.data

        with open(file_name, 'wb') as f:
            pickle.dump(params_dict, f)

    def load_params(self, file_name: str="params.pkl") -> None:
        with open(file_name, 'rb') as f:
            params_dict: dict[str, np.ndarray] = pickle.load(f)

        current_params: list[Parameter] = self.params()

        for i, param in enumerate(current_params):
            key: str = param.name if param.name else f"param_{i}"
            if key in params_dict:
                param.data = params_dict[key]

    def _get_convolution_outsize(self, h, w, fh, stride, pad) -> tuple[float, float]:
        out_h = (h - fh + 2 * pad) // stride + 1
        out_w = (w - fh + 2 * pad) // stride + 1
        return out_h, out_w
