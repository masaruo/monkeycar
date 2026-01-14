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
    
    def __init__(self):
        try:
            kit = ServoKit(channels=16)
            self._throttle = kit.continuous_servo[1]
            self._throttle.set_pulse_width_range(1000, 2000)
            self._steering = kit.servo[0]
            self._steering.set_pulse_width_range(1000, 2000)
            self.setup_esc()
            # self.deadzone = 0.01
            logger.info("MotorController initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MotorController: {e}")
            raise

    @property
    def throttle(self):
        return self._throttle.throttle

    @throttle.setter
    def throttle(self, raw_value):
        """
        raw_value: ジョイスティック(軸5)の -1.0(離す)～1.0(全開) 
                または AIの出力 0.0～1.0
        """
        # 1. ジョイスティックの [-1, 1] を [0, 1] に変換
        # 注意: AIの出力が元々 [0, 1] ならば、呼び出し側かここで判別が必要
        # ここでは「入力が負の値を取りうる」ことを前提に安全に処理する
        
        # もし axis 5 が 離して -1, 握って 1 なら以下の変換が必要
        # ただし、AI出力との一貫性を保つため、max(0, raw_value) を適用
        # 既に 0 が入力されている場合はそのまま 0 になる
        
        normalized = (raw_value + 1.0) / 2.0 if raw_value < 0 and raw_value >= -1.0 else raw_value
        
        # 2. デッドゾーンとクランプ
        if abs(normalized) < 0.01:
            final_throttle = 0.0
        else:
            # 物理的な最大・最小速度にクランプ
            final_throttle = max(0.0, min(normalized, MAX_SPEED))
        
        # 3. 実際のハードウェア(PCA9685)へ書き込み
        self._throttle.throttle = final_throttle

    @property
    def steering(self):
        return self._steering.angle

    @steering.setter
    def steering(self, value):
        angle = STEERING_CENTER + (value * 80)
        clamped = max(STEERING_CENTER - MAX_LEFT, min(angle, STEERING_CENTER + MAX_RIGHT))
        self._steering.angle = clamped

    def setup_esc(self):
        logger.info("Setting Up ESC")
        self.throttle = 0.0

    def set_speed(self, speed):
        """Set throttle speed.
        
        Args:
            speed: Float between -MAX_SPEED and MAX_SPEED
                   Negative = reverse, Positive = forward
        """
        self.throttle = speed

    def stop(self):
        """Stop the vehicle"""
        self.throttle = 0.0

    def turn_left(self, angle=45):
        """Turn left by angle degrees"""
        self.steering = -abs(angle)

    def turn_right(self, angle=45):
        """Turn right by angle degrees"""
        self.steering = abs(angle)

    def center_steering(self):
        """Center the steering wheel"""
        self.steering = 0

    def cleanup(self):
        """Safely stop vehicle and center steering."""
        logger.info("Cleaning up motor controller")
        self.stop()
        self.center_steering()

    def drive_test(self):
        """Test throttle and steering with safe values.
        
        Warning: Ensure vehicle wheels are off the ground before running!
        """
        logger.info("Starting drive test")
        
        print("微速前進... (slow forward)")
        self.throttle = 0.05
        time.sleep(2.0)
        self.stop()
 
        print("停止（ブレーキ） (stop/brake)")
        self.stop()
        time.sleep(1.0)

        print("左転 (turn left)")
        self.turn_left(50)
        time.sleep(1.0)
        
        print("右転 (turn right)")
        self.turn_right(50)
        time.sleep(1.0)
        
        print("センタリング (center)")
        self.center_steering()
        
        logger.info("Drive test completed")

if __name__ == "__main__":
    motor = Motor()
    motor.drive_test()
