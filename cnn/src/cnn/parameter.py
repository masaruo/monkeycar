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

    def __repr__(self):
        id_str = self.name if self.name else "Param"
        shape_str = str(self.data.shape) if self.data else "None"
        grad_status = "active" if self.grad else "inactive"

        return f"<{id_str} shape=[{shape_str} grad={grad_status}]"
