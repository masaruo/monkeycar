from typing import Self, Final
import logging
from adafruit_servokit import ServoKit


STOP_SPEED: Final = -1.0
THROTTLE_OFFSET: Final = 0.0
STEERING_TRIM: Final = -1
STEERING_CENTER: Final = 90 + STEERING_TRIM
MAX_LEFT: Final = 30
MAX_RIGHT: Final = 30

logger = logging.getLogger(__name__)

class Motor:
    def __init__(self, deadzone: float = 0.1):
        try:
            kit = ServoKit(channels=16)
            self._throttle = kit.continuous_servo[1]
            self._throttle.set_pulse_width_range(1500, 2500)
            self._steering = kit.servo[0]
            # self._steering.set_pulse_width_range(1000, 2000)
            self._steering.set_pulse_width_range(500, 2500)
            self._steer_angle = self._steering.angle
            self._deadzone = deadzone
            self.__setup_esc()
            logger.info("MotorController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            raise

    def accelerate(self, raw_value: float) -> None:
        """
        raw_value: -1.0 (停止) ～ 1.0 (全開)
        """
        # 1. 入力を [-1.0, 1.0] に制限
        val = max(-1.0, min(1.0, raw_value))
        self._throttle.throttle = val


    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        val = max(-1.0, min(1.0, raw_value))
        if val < 0:
            angle = STEERING_CENTER + (val * MAX_LEFT)
        else:
            angle = STEERING_CENTER + (val * MAX_RIGHT)
        final_angle = max(0, min(180, int(angle)))
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
        print("\n=== TAMIYA ESC CALIBRATION (-1.0 to 1.0 BASE) ===")
        # STEP 1: 前進ハイポイント (RT全開)
        print("\n[STEP 1] RTを全開(1.0)にしてEnter...")
        m._throttle.throttle = 1.0
        input(">> SETボタンを押し、Enter...")

        # STEP 2: バック端 (LTをダミーとして利用)
        # 停止位置(-1.0)よりさらに低い値を送り、ここをバック端にする
        print("\n[STEP 2] LTを全開にしてEnter (物理的に -2.0 を送信)...")
        m._throttle.throttle = -1.0 
        input(">> SETボタンを押し、Enter...")
