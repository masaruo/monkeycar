from typing import Self, Final
import logging
from adafruit_servokit import ServoKit

MAX_THROTTLE: Final = 0.5
DEAD_ZONE: Final = 0.02
STOP_SPEED: Final = 0.0
THROTTLE_OFFSET: Final = 0.0
STEERING_TRIM: Final = -15
STEERING_CENTER: Final = 90 + STEERING_TRIM
MAX_LEFT: Final = 40
MAX_RIGHT: Final = 30

logger = logging.getLogger(__name__)

class Motor:
    def __init__(self):
        try:
            kit = ServoKit(channels=16, frequency=50)
            self._throttle = kit.continuous_servo[1]
            self._throttle.set_pulse_width_range(1000, 2000)
            self._steering = kit.servo[0]
            self._steering.set_pulse_width_range(800, 2300)
            self._steer_angle = self._steering.angle
            self.__setup_esc()
            logger.info("MotorController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            raise

    # def accelerate(self, raw_value: float) -> None:
    #     """
    #     raw_value: -1.0 (停止) ～ 1.0 (全開)
    #     ESCは0.0を中立と考える。throttleの-0.9で前進してほしい
    #     -1.0 ~ 1.0を0.0 ~ 1.0に変えて送る
    #     """
    #     # -1.0 ~ 1.0に念のため
    #     clipped = max(-1.0, min(1.0, raw_value))

    #     # 0.0 ~ 1.0の枠に変換
    #     normalized = (clipped + 1) / 2
    #     output = max(0.0, min(1.0, normalized)) * MAX_THROTTLE
    #     output = 0 if output < DEAD_ZONE else output
    #     self._throttle.throttle = output

    # def accelerate(self, raw_value: float) -> None:
    #     """
    #     raw_value: 0.0 (停止) ～ 1.0 (全開)
    #     ※ AIのTanh出力が [-1, 1] であっても、0以下を停止とみなす設計
    #     """
    #     # 1. 0.0未満を切り捨て (バック不要のため)
    #     clipped = max(0.0, min(1.0, raw_value))

    #     # 2. 最大出力制限の適用
    #     output = clipped * MAX_THROTTLE
        
    #     # 3. デッドゾーン（物理的な動き出し閾値）の処理
    #     # 出力が小さすぎる場合は完全に 0 にしてサーボの唸りを防ぐ
    #     if output < DEAD_ZONE:
    #         final_output = 0.0
    #     else:
    #         final_output = output
            
    #     self._throttle.throttle = final_output
    def accelerate(self, raw_value: float) -> None:
        physical_val = (raw_value + 1.0) / 2.0
        final_val = physical_val * MAX_THROTTLE
        self._throttle.throttle = final_val

    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        val = max(-1.0, min(1.0, raw_value))

        #3乗して中心付近の過敏さを低減
        # val = val ** 3
        # if abs(val) < DEAD_ZONE:
        #     val = 0

        if val < 0:
            angle = STEERING_CENTER - (abs(val) * MAX_LEFT)
        else:
            angle = STEERING_CENTER + (val * MAX_RIGHT)
        final_angle = max(0, min(180, angle))
        self._steering.angle = final_angle

    def __setup_esc(self) -> None:
        logger.info("Setting Up ESC")
        self.stop()

    def stop(self):
        """Stop the vehicle"""
        self.accelerate(STOP_SPEED)
        self.steer(0)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        logger.info("motor stopped")
        return False
