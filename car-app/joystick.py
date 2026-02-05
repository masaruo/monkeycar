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

    def get_axis(self, index: int) -> float:
        """Return axis value in [-1.0, 1.0] with deadzone applied."""
        self.poll()
        return float(self.joystick.get_axis(index))

    def get_steering(self, axis_index: int = 0) -> float:
        """Horizontal stick axis in [-1.0, 1.0]; left negative, right positive."""
        if self.joystick:
            return self.get_axis(axis_index)
        else:
            return 0.0

    def get_throttle(self, axis_index: int = 5) -> float:
        """Vertical stick axis. If forward_only, map to [0.0, 1.0] (up = forward).

        Many controllers have up = -1 and down = +1. With forward_only=True,
        throttle = max(0, -axis_value). Otherwise returns raw in [-1.0, 1.0].
        """
        if self.joystick:
            val = self.get_axis(axis_index)
            return max(-1.0, min(1.0, val))
        else:
            return -1.0

    def get_throttle(self, axis_index: int = 5) -> float:
        """
        コントローラーの入力を取得し、0.0(停止) ～ 1.0(最大前進) に変換する。
        軸インデックスは環境に合わせて調整すること（通常は 5 または 2）。
        """
        if self.joystick:
            # 生の値を取得 (通常は離すと -1.0, 押し込むと 1.0)
            raw_val = self.get_axis(axis_index)
            
            # [-1.0, 1.0] -> [0.0, 1.0] への線形変換
            # これをこのシステムにおける「Rawスロットル」として扱う
            throttle = (raw_val + 1.0) / 2.0
            
            # 浮動小数点の誤差や微小な入力を防ぐためのガード
            return max(0.0, min(1.0, throttle))
            
        return 0.0

    def get_button(self, index: int) -> bool:
        if self.joystick:
            self.poll()
            return bool(self.joystick.get_button(index))
        else:
            return False
