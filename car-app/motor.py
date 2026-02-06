from typing import Self, Final
import logging
from adafruit_servokit import ServoKit

SPPED_REDUCTION_RATIO: Final = 0.5 #! MO DO NOT CHANGE
DEAD_ZONE: Final = 0.01
PHISICAL_START_SPEED = 0.10
STOP_SPEED: Final = 0.0

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

    def accelerate(self, raw_value: float) -> None:
        """
        入力値 -1.0 ~ 1.0
        """
        # 推論などで0.0以下や、1.0以上の数字が入ってきたら範囲内に転換
        clamped = max(0.0, min(1.0, raw_value))
        logging.info(f"raw_throttle:{clamped}")
        final_throttle = clamped
        if clamped < DEAD_ZONE:
            final_throttle = 0.0
        else:
            effective_range = 1.0 - PHISICAL_START_SPEED
            final_throttle = PHISICAL_START_SPEED + (clamped * SPPED_REDUCTION_RATIO * effective_range)
        self._throttle.throttle = final_throttle

    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        val = max(-1.0, min(1.0, raw_value))

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
