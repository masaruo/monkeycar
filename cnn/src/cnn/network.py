import numpy as np
import logging
from .layers import (
    Affine,
    Relu,
    Convolution,
    Pooling,
    Flatten,
    MeanSquaredError,
    Dropout,
)
import pickle
from typing import Union
from .parameter import Parameter

logger = logging.getLogger(__name__)


class CarConvNet:
    """
    3層の畳み込み層を持つCNN (Parameterクラス対応版)
    構成: Conv -> Relu -> Pool (x3) -> Flatten -> Affine -> Relu -> Dropout -> Affine -> MSE
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int] = (3, 120, 160),
        hidden_size: int = 100,
        output_size: int = 2,
    ) -> None:
        input_channels, input_h, input_w = input_dim

        self.Convlayers = [
            # Conv1 (out_channleはハイパーパラメーター)
            Convolution(
                in_channels=input_channels,
                out_channels=16,
                filter_size=3,
                stride=2,
                pad=1,
                name="Conv1",
            ),
            Relu(),
            Pooling(pool_h=2, pool_w=2, stride=2),
            # Conv2
            Convolution(
                in_channels=16,
                out_channels=32,
                filter_size=3,
                stride=1,
                pad=0,
                name="Conv2",
            ),
            Relu(),
            Pooling(pool_h=2, pool_w=2, stride=2),
            # Conv3
            Convolution(
                in_channels=32,
                out_channels=64,
                filter_size=3,
                stride=1,
                pad=0,
                name="Conv3",
            ),
            Relu(),
            Pooling(pool_h=2, pool_w=2, stride=2),
        ]
        flatten_dim = self.__get_flatten_dim(input_dim=input_dim)
        self.AffineLayers = [
            Flatten(),
            Affine(input_size=flatten_dim, output_size=hidden_size, name="Affine1"),
            Relu(),
            Dropout(0.5),
            Affine(input_size=hidden_size, output_size=output_size, name="Affine2"),
        ]
        self.layers = self.Convlayers + self.AffineLayers
        self.last_layer = MeanSquaredError()

    def __get_flatten_dim(self, input_dim) -> int:
        dummy_x: np.ndarray = np.zeros((1, *input_dim))
        for layer in self.Convlayers:
            dummy_x = layer.forward(dummy_x)
        return dummy_x.size

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

        # 2. Backward
        dout: Union[float, np.ndarray] = 1.0
        dout = self.last_layer.backward(dout)

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return None

    def params(self) -> list[Parameter]:
        all_params: list[Parameter] = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                all_params.extend(layer.parameters())
        return all_params

    def train_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval_mode(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def save_params(self, file_name: str = "params.pkl") -> None:
        params_dict: dict[str, np.ndarray] = {}

        for i, param in enumerate(self.params()):
            key: str = param.name if param.name else f"param_{i}"
            params_dict[key] = param.data

        with open(file_name, "wb") as f:
            pickle.dump(params_dict, f)

    def load_params(self, file_name: str = "params.pkl") -> None:
        with open(file_name, "rb") as f:
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
