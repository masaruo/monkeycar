from typing import Self, Final
import logging
from adafruit_servokit import ServoKit
# api ref https://docs.circuitpython.org/projects/servokit/en/latest/api.html

# from shared.config import MAX_SPEED, MAX_LEFT, MAX_RIGHT, STEERING_CENTER, STOP_SPEED

SPEED_REDUCTION_RATIO: Final = 0.01
STOP_SPEED: Final = -1.0
STEERING_TRIM: Final = -10
STEERING_CENTER: Final = 90 + STEERING_TRIM
MAX_LEFT: Final = 50
MAX_RIGHT: Final = 50

logger = logging.getLogger(__name__)

class Motor:
    """Controls motor throttle and steering for the vehicle.
    
    Manages ESC (Electronic Speed Controller) and steering servo.
    Throttle range: 0 (stop) to MAX_SPEED (forward only, no reverse)
    Steering range: -MAX_LEFT to +MAX_RIGHT degrees from center
    """
    
    def __init__(self, deadzone: float = 0.1):
        try:
            kit = ServoKit(channels=16)
            self._throttle = kit.continuous_servo[1]
            self._throttle.set_pulse_width_range(500, 2500)
            self._steering = kit.servo[0]
            self._steering.set_pulse_width_range(500, 2500)
            self._steer_angle = self._steering.angle
            self._deadzone = deadzone
            self.__setup_esc()
            logger.info("MotorController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            raise

    def accelerate(self, raw_value: float) -> None:
        # raw_value: -1.0 (stop) to +1.0 (full forward)
        # Apply deadzone around stop point (-1.0)
        normalized: float = (1 + raw_value) / 2 # 0.0 ~ 1.0
        if normalized < self._deadzone:
            throttle = 0.0 #! NO REVERSE
        else:
            throttle = normalized * SPEED_REDUCTION_RATIO
        final_throttle = min(throttle, 1.0)
        self._throttle.throttle = final_throttle

    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        if raw_value < 0:
            angle = STEERING_CENTER + (raw_value * MAX_LEFT)
        else:
            angle = STEERING_CENTER + (raw_value * MAX_RIGHT)
        self._steering.angle = int(angle)

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
    print("print calib")
    import time
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main_logger = logging.getLogger()
    main_logger.info("calibration started")
    with Motor() as m:
        main_logger.info(f"neutral")
        m.steer(0)
        time.sleep(2)

        main_logger.info(f"left")
        m.steer(-1.0)
        time.sleep(2)

        main_logger.info(f"right")
        m.steer(1.0)
        time.sleep(2)

        main_logger.info(f"neutral")
        m.steer(0)
        time.sleep(2)
