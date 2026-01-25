import logging
import tensorflow as tf
import keras
from keras import layers


logger = logging.getLogger(__name__)

class CarModel:
    """軽量CNN モデル (ラズパイで動作可能)"""
    
    @staticmethod
    def build_model(
        input_shape: tuple = (160, 120, 3),
        output_type: str = 'continuous'  # 'continuous' または 'classification'
    ) -> keras.Model:
        """
        入力: カメラ画像
        出力: ステアリング値 + スロットル値
        
        Args:
            input_shape: 入力画像のシェイプ (高さ, 幅, チャンネル)
            output_type: 出力タイプ ('continuous'=回帰, 'classification'=分類)
        """
        model = keras.Sequential([
            # 入力層
            layers.Input(shape=input_shape),
            
            # データ拡張 (訓練時のみ)
            # layers.RandomFlip("horizontal"),
            # layers.RandomRotation(0.1),
            
            # CNN ブロック 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # CNN ブロック 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # CNN ブロック 3
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # 全結合層
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            
            # 出力層 (マルチタスク学習)
            # - steering: [-1, 1] を出力
            # - throttle: [0, 1] を出力
            layers.Dense(2, name='output'),  # [steering, throttle]
        ])
        
        return model
