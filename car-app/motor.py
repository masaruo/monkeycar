from typing import Self, Final
import logging
from adafruit_servokit import ServoKit

MAX_THROTTLE: Final = 0.5
DEAD_ZONE: Final = 0.05
STOP_SPEED: Final = -1.0
THROTTLE_OFFSET: Final = 0.0
STEERING_TRIM: Final = -1
STEERING_CENTER: Final = 90 + STEERING_TRIM
MAX_LEFT: Final = 30
MAX_RIGHT: Final = 30

logger = logging.getLogger(__name__)

class Motor:
    def __init__(self):
        try:
            kit = ServoKit(channels=16, frequency=50)
            self._throttle = kit.continuous_servo[1]
            self._throttle.set_pulse_width_range(500, 2500)
            self._steering = kit.servo[0]
            self._steering.set_pulse_width_range(800, 2200)
            self._steer_angle = self._steering.angle
            self.__setup_esc()
            logger.info("MotorController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            raise

    def accelerate(self, raw_value: float) -> None:
        """
        raw_value: -1.0 (停止) ～ 1.0 (全開)
        ESCは0.0を中立と考える。throttleの-0.9で前進してほしい
        -1.0 ~ 1.0を0.0 ~ 1.0に変えて送る
        """
        # -1.0 ~ 1.0に念のため
        clipped = max(-1.0, min(1.0, raw_value))

        # 0.0 ~ 1.0の枠に変換
        normalized = (clipped + 1) / 2
        output = max(0.0, min(1.0, normalized)) * MAX_THROTTLE
        output = 0 if output < DEAD_ZONE else output
        self._throttle.throttle = output

    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        val = max(-1.0, min(1.0, raw_value))

        #3乗して中心付近の過敏さを低減
        val = val ** 3
        if abs(val) < DEAD_ZONE:
            val = 0

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

if __name__ == "__main__":
    with Motor() as m:
        print("\n=== TAMIYA ESC CALIBRATION (正しい手順) ===")

        # ---------------------------------------------------------
        # 【重要】 STEP 1: ニュートラル (0.0)
        # ---------------------------------------------------------
        m._throttle.throttle = 0.0
        # ★ここで止まるのが大事！今は「停止」の信号を送っています。
        print("★ ESCを「設定モード（点滅）」にしてから Enter を押してください")
        input(">> LEDが変わったら Enter...")

        # ---------------------------------------------------------
        # STEP 2: 前進 (1.0)
        # ---------------------------------------------------------
        print("\n[STEP 2] 前進ハイポイント(1.0) を送信中...")
        m._throttle.throttle = 1.0
        input(">> SETボタンを「1回」押し、LEDが変わったら Enter...")

        # ---------------------------------------------------------
        # STEP 3: 後退 (-1.0)
        # ---------------------------------------------------------
        print("\n[STEP 3] バックハイポイント(-1.0) を送信中...")
        m._throttle.throttle = -1.0 
        input(">> SETボタンを「1回」押し、LEDが消灯したら Enter...")

        # 終了
        m._throttle.throttle = 0.0
        print("\n[完了] すべて終了です！")
