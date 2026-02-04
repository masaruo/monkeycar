import pygame
import os
import logging


logger = logging.getLogger(__name__)


# SSH経由などディスプレイがない環境（ヘッドレス）でのエラー回避
os.environ["SDL_VIDEODRIVER"] = "dummy"

class Joystick:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick = None

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            # Propagate exceptions if any
            return False

    def close(self):
        try:
            if self.joystick and self.joystick.get_init():
                self.joystick.quit()
        except Exception:
            pass
        pygame.quit()
        logger.info("Joystick closed and pygame quit")

    def poll(self):
        """Pump pygame events to keep the joystick state updated."""
        if self.joystick:
            pygame.event.pump()

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float = 0.05) -> float:
        return 0.0 if abs(value) < deadzone else value

    def get_axis(self, index: int, deadzone: float = 0.05) -> float:
        """Return axis value in [-1.0, 1.0] with deadzone applied."""
        self.poll()
        raw = self.joystick.get_axis(index)
        return self._apply_deadzone(raw, deadzone)

    def get_steering(self, axis_index: int = 0, deadzone: float = 0.05) -> float:
        """Horizontal stick axis in [-1.0, 1.0]; left negative, right positive."""
        if self.joystick:
            return self.get_axis(axis_index, deadzone)
        else:
            return 0.0

    def get_throttle(self, axis_index: int = 5, deadzone: float = 0.05) -> float:
        """Vertical stick axis. If forward_only, map to [0.0, 1.0] (up = forward).

        Many controllers have up = -1 and down = +1. With forward_only=True,
        throttle = max(0, -axis_value). Otherwise returns raw in [-1.0, 1.0].
        """
        if self.joystick:
            val = self.get_axis(axis_index, deadzone)
            return val
        else:
            return -1.0

    def get_button(self, index: int) -> bool:
        if self.joystick:
            self.poll()
            return bool(self.joystick.get_button(index))
        else:
            return False
