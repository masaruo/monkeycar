import numpy as np
from cnn.parameter import Parameter
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lr: float):
        self.lr = lr

    @abstractmethod
    def update(self, params: list[Parameter]) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001):
        self.lr = lr

    def update(self, params: list[Parameter]) -> None:
        for param in params:
            if param.grad is None:
                continue

            param.data -= self.lr * param.grad


class Adam(Optimizer):
    """Adam p175"""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0

        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}

    def update(self, params: list[Parameter]) -> None:
        self.iter += 1

        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2**self.iter)
            / (1.0 - self.beta1**self.iter)
        )

        for param in params:
            if param.grad is None:
                continue

            key = id(param)

            # 初回は０で初期化
            if key not in self.m:
                self.m[key] = np.zeros_like(param.data)
                self.v[key] = np.zeros_like(param.data)

            # モメンタムの更新
            self.m[key] += (1 - self.beta1) * (param.grad - self.m[key])
            # 速度の更新
            self.v[key] += (1 - self.beta2) * (param.grad**2 - self.v[key])
            # パラメーターの更新
            param.data -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
