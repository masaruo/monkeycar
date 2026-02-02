from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(eq=False)
class Parameter:
    """
    重みと勾配を管理するクラス
    """
    data: np.ndarray
    grad: Optional[np.ndarray] = None
    name: Optional[str] = None

    @property
    def shape(self) -> tuple:
        return self.data.shape
