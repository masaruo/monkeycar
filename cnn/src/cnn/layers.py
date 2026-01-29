import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Self

logger = logging.getLogger(__name__)


class Layer(ABC):
    @abstractmethod
    def forward(self: Self, x:np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self: Self, dout:np.ndarray) -> np.ndarray:
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
