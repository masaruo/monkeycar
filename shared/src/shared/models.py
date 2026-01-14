from pydantic import BaseModel, Field
import time

class DriveData(BaseModel):
    """ラズパイで収集される1ステップ分の走行データ"""
    timestamp: float = Field(default_factory=time.time)
    image_name: str = Field(..., description="path to a corresponding image file")
    steering: float = Field(..., ge=-1.0, le=1.0)
    throttle: float = Field(..., ge=-1.0, le=1.0)

# class InferenceResult(BaseModel):
#     """FTLiteモデルからの出力データ"""
