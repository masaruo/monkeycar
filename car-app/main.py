import time
import logging
from motor import Motor
from joystick import Joystick
from camera import Camera
from recorder import Recorder
from inference import Inference
import argparse
from typing import Final

# Joystick Settings
A: Final = 0
B: Final = 1
X: Final = 2
Y: Final = 3
RT: Final = 5
LEFTSTICK: Final = 0

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autopilot", action='store_true')
    parser.add_argument("--record", action='store_true')
    args = parser.parse_args()

    # recorder = Recorder(base_dir="data")
    try:
        pilot = Inference("./output")
    except Exception:
        pilot = None

    is_auto: bool = args.autopilot
    is_recording: bool = args.record

    prev_x_pressed: bool = False
    prev_b_pressed = False

    with Joystick() as js, Motor() as motor, Camera() as cam:
        try:
            while True:
                js.poll()
                x_pressed = js.get_button(X) #! X = autopilot
                if x_pressed and not prev_x_pressed:
                    logger.info(f"AutoPilot mode {'stop' if is_auto else 'start'}")
                    is_auto = not is_auto
                prev_x_pressed = x_pressed

                b_pressed = js.get_button(B) #! B = record
                if b_pressed and not prev_b_pressed:
                    if not is_recording:
                        recorder = Recorder(base_dir="./data")

                    logger.info(f"Recording {'stop' if is_recording else 'start'}")
                    is_recording = not is_recording
                prev_b_pressed = b_pressed

                if js.get_button(Y):
                    raise KeyboardInterrupt("Y button pressed")

                frame = cam.capture()

                steer: float
                throttle: float

                if is_auto and pilot:
                    steer, throttle = pilot.predict(frame=frame)
                else:
                    steer = js.get_steering(axis_index=LEFTSTICK)# -1.0～+1 .0
                    throttle = js.get_throttle(axis_index=RT)# -1.0~+1.0

                # スケーリングは環境に合わせて調整
                motor.steer(steer)
                motor.accelerate(throttle)

                if is_recording:
                    recorder.save(frame, steer, throttle)

                # logger.info(f"steering:[{steer}] and throttle:[{throttle}]")
                time.sleep(0.02)
        except KeyboardInterrupt:
            logger.info("Finish main loop")

if __name__ == "__main__":
    main()
