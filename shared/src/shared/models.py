from pydantic import BaseModel, Field
import time

class DriveData(BaseModel):
    """ラズパイで収集される1ステップ分の走行データ"""
    timestamp: float = Field(default_factory=time.time)
    image_name: str = Field(..., description="path to a corresponding image file")
    steering: float = Field(..., ge=-1.0, le=1.0)
    throttle: float = Field(..., ge=-1.0, le=1.0)

from pydantic import BaseModel, Field
from typing import List

class ModelConfig(BaseModel):
    # 推論・前処理用
    image_size: List[int] = Field(..., description="[Width, Height]")
    image_shape: List[int] = Field(description="[Height,  Width, Channels]")
    
    # 制御値の正規化用
    steering_min: float = -1.0
    steering_max: float = 1.0
    throttle_min: float = -1.0
    throttle_max: float = 1.0
    
    # 訓練メタデータ (Optional)
    num_samples: int
    epochs_trained: int
    final_loss: float
    final_val_loss: float
