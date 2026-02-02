from pydantic import BaseModel, Field
import time

class DriveData(BaseModel):
    """ラズパイで収集される1ステップ分の走行データ"""
    timestamp: float = Field(default_factory=time.time)
    image_name: str = Field(..., description="path to a corresponding image file")
    steering: float = Field(..., ge=-1.0, le=1.0)
    throttle: float = Field(..., ge=-1.0, le=1.0)


class ModelConfig(BaseModel):
    # 推論・前処理用
    image_size: tuple[int, int] = Field(default=(160, 120), description="[Width, Height]")
    norm_divisor: float = Field(default=255.0, gt=0, description="Normalization divisor")
    image_shape: tuple[int, int, int] | None = Field(default=None, description="[Height, Width, Channels]")
    
    # 制御値の正規化用 - 訓練後に設定される（デフォルト値なし）
    steering_min: float | None = Field(default=None, description="Min steering value")
    steering_max: float | None = Field(default=None, description="Max steering value")
    throttle_min: float | None = Field(default=None, description="Min throttle value")
    throttle_max: float | None = Field(default=None, description="Max throttle value")
    
    # 訓練メタデータ
    num_samples: int | None = Field(default=None, description="Number of training samples")
    final_loss: float | None = Field(default=None, description="Final training loss")
    final_val_loss: float | None = Field(default=None, description="Final validation loss")
