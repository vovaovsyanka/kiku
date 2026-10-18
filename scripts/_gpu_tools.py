"""Общие утилиты скриптов диагностики: опрос nvidia-smi во время работы."""
import shutil
import subprocess
import sys
import threading
import time


class GpuSampler:
    """Раз в ~0.25 с спрашивает nvidia-smi о загрузке GPU и памяти (если nvidia-smi доступен)."""
    def __init__(self):
        self.ok = shutil.which("nvidia-smi") is not None
        self.util: list[int] = []
        self.mem: list[int] = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        flags = 0x08000000 if sys.platform.startswith("win") else 0   # без мигающих окон
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, creationflags=flags).stdout.strip().splitlines()[0]
                u, m = [int(x.strip()) for x in out.split(",")]
                self.util.append(u)
                self.mem.append(m)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.25)

    def __enter__(self):
        if self.ok:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> str:
        if not self.ok:
            return "nvidia-smi недоступен — загрузку GPU измерить нельзя"
        if not self.util:
            return "нет замеров nvidia-smi"
        return (f"загрузка GPU: в среднем {sum(self.util) / len(self.util):.0f}%, максимум {max(self.util)}%; "
                f"видеопамять до {max(self.mem)} МБ ({len(self.util)} замеров)")
