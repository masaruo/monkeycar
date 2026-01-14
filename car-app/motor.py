import time
import logging
from adafruit_servokit import ServoKit

from config import MAX_SPEED, MAX_LEFT, MAX_RIGHT, STEERING_CENTER, MIN_SPEED


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
            self._throttle.set_pulse_width_range(1000, 2000)
            self._steering = kit.servo[0]
            self._steering.set_pulse_width_range(1000, 2000)
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
        if abs(raw_value - (-1.0)) < self._deadzone:
            final_throttle = -1.0
        else:
            final_throttle = max(-1.0, min(raw_value, MAX_SPEED))
        self._throttle.throttle = final_throttle

    def steer(self, raw_value: float) -> None:
        # raw_value = -1.0 ~ 1.0
        # Scale by MAX_LEFT/MAX_RIGHT (both 50) to get full range
        angle = STEERING_CENTER + (raw_value * MAX_LEFT)
        clamped = max(STEERING_CENTER - MAX_LEFT, min(angle, STEERING_CENTER + MAX_RIGHT))
        self._steering.angle = int(clamped)

    def __setup_esc(self):
        logger.info("Setting Up ESC")
        self.stop()

    def stop(self):
        """Stop the vehicle"""
        self.accelerate(-1.0)

#     def drive_test(self):
#         """Test throttle and steering with safe values.
        
#         Warning: Ensure vehicle wheels are off the ground before running!
#         """
#         logger.info("Starting drive test")
        
#         print("微速前進... (slow forward)")
#         self.throttle = 0.05
#         time.sleep(2.0)
#         self.stop()
 
#         print("停止（ブレーキ） (stop/brake)")
#         self.stop()
#         time.sleep(1.0)

#         print("左転 (turn left)")
#         self.turn_left(50)
#         time.sleep(1.0)
        
#         print("右転 (turn right)")
#         self.turn_right(50)
#         time.sleep(1.0)
        
#         print("センタリング (center)")
#         self.center_steering()
        
#         logger.info("Drive test completed")

# if __name__ == "__main__":
#     motor = Motor()
#     motor.drive_test()
