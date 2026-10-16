"""Окна, громкость, медиа-клавиши, блокировка экрана (только Windows, чистый ctypes — без доп. пакетов).

Окна ищутся по заголовку. Разговорные названия («вс код», «хром») сопоставляются с заголовками
через config.yaml → windows.aliases.
"""
import sys
import time
from typing import Optional

from .registry import ToolContext, ToolError, ToolRegistry

SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE = 3, 6, 9
VK = {"volume_up": 0xAF, "volume_down": 0xAE, "volume_mute": 0xAD,
      "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1}


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("ё", "е").split())


class UnsupportedBackend:
    """Заглушка для не-Windows систем (и для тестов без бэкенда)."""
    def _no(self, *a, **k):
        raise ToolError("управление окнами и звуком работает только в Windows")
    list_windows = show = focus = close = press_key = lock = _no


class Win32Backend:
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w

        self.ct = ctypes
        u = ctypes.WinDLL("user32", use_last_error=True)
        self.WNDENUMPROC = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        u.EnumWindows.argtypes = [self.WNDENUMPROC, w.LPARAM]
        u.EnumWindows.restype = w.BOOL
        u.IsWindowVisible.argtypes = [w.HWND]
        u.IsWindowVisible.restype = w.BOOL
        u.GetWindowTextLengthW.argtypes = [w.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.restype = ctypes.c_int
        u.GetWindowLongW.argtypes = [w.HWND, ctypes.c_int]
        u.GetWindowLongW.restype = ctypes.c_long
        u.GetWindow.argtypes = [w.HWND, w.UINT]
        u.GetWindow.restype = w.HWND
        u.ShowWindow.argtypes = [w.HWND, ctypes.c_int]
        u.ShowWindow.restype = w.BOOL
        u.IsIconic.argtypes = [w.HWND]
        u.IsIconic.restype = w.BOOL
        u.SetForegroundWindow.argtypes = [w.HWND]
        u.SetForegroundWindow.restype = w.BOOL
        u.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        u.PostMessageW.restype = w.BOOL
        u.keybd_event.argtypes = [w.BYTE, w.BYTE, w.DWORD, ctypes.c_size_t]
        u.keybd_event.restype = None
        u.LockWorkStation.argtypes = []
        u.LockWorkStation.restype = w.BOOL
        self.u = u
        self.dwm = ctypes.WinDLL("dwmapi")
        self.dwm.DwmGetWindowAttribute.argtypes = [w.HWND, w.DWORD, ctypes.c_void_p, w.DWORD]

    def list_windows(self) -> list[tuple[int, str]]:
        ct, u = self.ct, self.u
        out: list[tuple[int, str]] = []

        def cb(hwnd, lparam):
            try:
                if not u.IsWindowVisible(hwnd):
                    return True
                n = u.GetWindowTextLengthW(hwnd)
                if n == 0:
                    return True
                buf = ct.create_unicode_buffer(n + 1)
                u.GetWindowTextW(hwnd, buf, n + 1)
                title = buf.value.strip()
                if not title or title == "Program Manager":
                    return True
                ex = u.GetWindowLongW(hwnd, -20)           # GWL_EXSTYLE
                if (ex & 0x80) and not (ex & 0x40000):     # TOOLWINDOW без APPWINDOW
                    return True
                if u.GetWindow(hwnd, 4):                   # GW_OWNER: дочерние диалоги пропускаем
                    return True
                cloaked = ct.c_int(0)                      # скрытые окна UWP
                self.dwm.DwmGetWindowAttribute(hwnd, 14, ct.byref(cloaked), ct.sizeof(cloaked))
                if cloaked.value:
                    return True
                out.append((int(hwnd or 0), title))
            except Exception:  # noqa: BLE001
                pass
            return True

        u.EnumWindows(self.WNDENUMPROC(cb), 0)
        return out

    def show(self, hwnd: int, cmd: int) -> None:
        self.u.ShowWindow(hwnd, cmd)

    def focus(self, hwnd: int) -> None:
        if self.u.IsIconic(hwnd):
            self.u.ShowWindow(hwnd, SW_RESTORE)
        self.u.keybd_event(0x12, 0, 0, 0)       # короткое нажатие Alt снимает запрет на смену активного окна
        self.u.keybd_event(0x12, 0, 2, 0)
        self.u.SetForegroundWindow(hwnd)

    def close(self, hwnd: int) -> None:
        self.u.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE: вежливое закрытие, программа может спросить о сохранении

    def press_key(self, vk: int) -> None:
        self.u.keybd_event(vk, 0, 0x1, 0)        # KEYEVENTF_EXTENDEDKEY
        self.u.keybd_event(vk, 0, 0x1 | 0x2, 0)  # + KEYUP

    def lock(self) -> None:
        self.u.LockWorkStation()


def register_window_tools(reg: ToolRegistry, ctx: ToolContext, backend=None) -> None:
    if backend is None:
        backend = UnsupportedBackend()
        if sys.platform.startswith("win"):
            try:
                backend = Win32Backend()
            except Exception:  # noqa: BLE001
                pass

    groups = []
    win_cfg = getattr(ctx.cfg, "windows", None)
    for a in (getattr(win_cfg, "aliases", None) or []):
        groups.append((_norm(a.title), {_norm(n) for n in a.names}))

    def resolve(query: str) -> str:
        q = _norm(query)
        for title, names in groups:
            if q == title or q in names:
                return title
        return q

    def find(query: str) -> list[tuple[int, str]]:
        q = resolve(query)
        return [(h, t) for h, t in backend.list_windows() if q in _norm(t)]

    def short(t: str) -> str:
        return t if len(t) <= 70 else t[:67] + "…"

    @reg.tool("list_windows", "Список открытых окон (заголовки).", reads=True)
    def list_windows() -> list:
        return [short(t) for _, t in backend.list_windows()][:25]

    @reg.tool("window_action",
              "Действие над окном по названию. action: minimize|maximize|restore|focus|close. "
              "title — часть заголовка или разговорное имя («вс код», «хром»).")
    def window_action(action: str, title: str) -> str:
        action = action.strip().lower()
        if action not in ("minimize", "maximize", "restore", "focus", "close"):
            raise ToolError("action: minimize, maximize, restore, focus или close")
        matches = find(title)
        if not matches:
            raise ToolError(f"окно не найдено: {title}")
        if action != "minimize" and len(matches) > 1:
            raise ToolError("подходит несколько окон: " + "; ".join(short(t) for _, t in matches) + " — уточните название")
        if action == "close" and not ctx.confirm(
                f"Закрыть окно «{short(matches[0][1])}»? Несохранённые данные могут пропасть."):
            raise ToolError("хозяин не подтвердил закрытие")
        names = ", ".join(short(t) for _, t in matches)
        if ctx.dry_run:
            return f"[dry-run] {action}: {names}"
        for hwnd, _ in matches:
            if action == "minimize":
                backend.show(hwnd, SW_MINIMIZE)
            elif action == "maximize":
                backend.show(hwnd, SW_MAXIMIZE)
            elif action == "restore":
                backend.show(hwnd, SW_RESTORE)
            elif action == "focus":
                backend.focus(hwnd)
            else:
                backend.close(hwnd)
        return f"{action}: {names}"

    @reg.tool("minimize_all", "Свернуть все окна (показать рабочий стол).")
    def minimize_all() -> str:
        wins = backend.list_windows()
        if ctx.dry_run:
            return f"[dry-run] свернула бы окон: {len(wins)}"
        for hwnd, _ in wins:
            backend.show(hwnd, SW_MINIMIZE)
        return f"свёрнуто окон: {len(wins)}"

    @reg.tool("volume", "Громкость системы. action: up|down|mute; steps — число шагов (по два процента), по умолчанию 5.")
    def volume(action: str, steps: int = 5) -> str:
        action = action.strip().lower()
        key = {"up": "volume_up", "down": "volume_down", "mute": "volume_mute"}.get(action)
        if not key:
            raise ToolError("action: up, down или mute")
        steps = 1 if key == "volume_mute" else max(1, min(int(steps), 25))
        if ctx.dry_run:
            return f"[dry-run] громкость {action} x{steps}"
        for _ in range(steps):
            backend.press_key(VK[key])
            time.sleep(0.02)
        return f"громкость: {action}"

    @reg.tool("media", "Медиа-клавиши. action: play_pause|next|previous.")
    def media(action: str) -> str:
        action = action.strip().lower()
        if action not in ("play_pause", "next", "previous"):
            raise ToolError("action: play_pause, next или previous")
        if ctx.dry_run:
            return f"[dry-run] медиа {action}"
        backend.press_key(VK[action])
        return f"медиа: {action}"

    @reg.tool("lock_screen", "Заблокировать компьютер (нужно подтверждение).")
    def lock_screen() -> str:
        if not ctx.confirm("Заблокировать компьютер?"):
            raise ToolError("хозяин не подтвердил")
        if ctx.dry_run:
            return "[dry-run] заблокировала бы экран"
        backend.lock()
        return "экран заблокирован"
