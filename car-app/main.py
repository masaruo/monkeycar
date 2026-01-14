import time
import logging
from motor import Motor
from joystick import Joystick
from camera import Camera
from recorder import Recorder
from interpreter import Interpreter

A: int = 0
B: int = 1
X: int = 2
Y: int = 3


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    camera = Camera()
    recorder = Recorder(base_dir="data")
    motor = Motor()
    pilot = Interpreter("./output/model.tflite", "./output/config.json")

    is_auto: bool = False
    is_recording: bool = False
    
    prev_x_pressed: bool = False
    prev_b_pressed = False

    with Joystick() as js:
        try:
            while True:
                js.poll()  # 1フレームにつき1回だけイベント更新
                x_pressed = js.get_button(X)
                if x_pressed and not prev_x_pressed:
                    logging.info(f"AutoPilot mode {'stop' if is_auto else 'start'}")
                    is_auto = not is_auto
                prev_x_pressed = x_pressed

                b_pressed = js.get_button(B)
                if b_pressed and not prev_b_pressed:
                    logging.info(f"Recording {'stop' if is_recording else 'start'}")
                    is_recording = not is_recording
                prev_b_pressed = b_pressed

                if js.get_button(Y):
                    raise KeyboardInterrupt("Y button pressed")
                
                frame = camera.capture()

                steer: float
                throttle: float

                if is_auto:
                    steer, throttle = pilot.predict(frame=frame)
                else:
                    steer = js.get_steering(axis_index=0)      # -1.0～+1.0
                    throttle = js.get_throttle(axis_index=5)

                # スケーリングは環境に合わせて調整
                motor.steering = steer
                motor.throttle = throttle

                if is_recording:
                    recorder.save(frame, steer, throttle) #実際のsteerと機械学習のステアリングが違う

                logging.info(f"steering:[{steer}] and throttle:[{throttle}]")
                time.sleep(0.02)
        except KeyboardInterrupt:
            motor.stop()
            camera.close()
            logger.info("Finish main loop")

if __name__ == "__main__":
    main()
