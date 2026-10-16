"""Политика доступа к файловой системе: Кику видит только разрешённые папки."""
import re
import sys
from pathlib import Path

from .registry import ToolError

# алиас -> (имя папки по умолчанию, GUID известной папки Windows)
_KNOWN = {
    "desktop":   ("Desktop",   "B4BFCC3A-DB2C-424C-B029-7FE99A87C641"),
    "documents": ("Documents", "FDD39AD0-238F-46AF-ADB4-6C85480369C7"),
    "downloads": ("Downloads", "374DE290-123F-4565-9164-39C4925E467B"),
    "pictures":  ("Pictures",  "33E28130-4E1E-4676-835A-98395C3BC3BB"),
    "music":     ("Music",     "4BD8D571-6D19-48D3-BE97-422220080E43"),
    "videos":    ("Videos",    "18989B1D-99B5-455B-841C-AB7C74E4DDFC"),
}

_RU = {
    "рабочий стол": "desktop", "рабочем столе": "desktop", "рабочего стола": "desktop",
    "стол": "desktop", "десктоп": "desktop",
    "документы": "documents", "документах": "documents", "документов": "documents",
    "мои документы": "documents",
    "загрузки": "downloads", "загрузках": "downloads", "загрузок": "downloads",
    "скачанное": "downloads", "download": "downloads",
    "картинки": "pictures", "изображения": "pictures", "фото": "pictures",
    "музыка": "music", "видео": "videos",
    "рабочая папка": "workspace", "workspace": "workspace",
}


def is_folder_alias(text: str) -> bool:
    """Это название стандартной папки («загрузки», «рабочий стол», «documents»)?"""
    t = (text or "").strip().lower().replace("ё", "е")
    return _RU.get(t, t) in _KNOWN or t == "workspace"


def _win_known_folder(guid_str: str):
    """Реальный путь системной папки Windows (учитывает OneDrive и перенос на другой диск)."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    u = uuid.UUID(guid_str)
    g = GUID(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))
    buf = ctypes.c_wchar_p()
    if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(buf)) != 0:
        return None
    path = buf.value
    ctypes.windll.ole32.CoTaskMemFree(buf)
    return Path(path) if path else None


def known_folder(alias: str) -> Path:
    name, guid = _KNOWN[alias]
    if sys.platform.startswith("win"):
        try:
            p = _win_known_folder(guid)
            if p:
                return p
        except Exception:  # noqa: BLE001
            pass
    return Path.home() / name


class PathPolicy:
    def __init__(self, cfg):
        sec = cfg.security
        self.known = {a: known_folder(a) for a in _KNOWN}
        self.workspace = Path(sec.workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.known["workspace"] = self.workspace
        self.roots: list[Path] = []
        for r in sec.allowed_roots:
            if r.startswith("@"):
                key = r[1:].lower()
                if key not in self.known:
                    raise ValueError(f"Неизвестная системная папка в allowed_roots: {r}")
                p = self.known[key]
            else:
                p = Path(r).expanduser()
            self.roots.append(p.resolve(strict=False))
        if self.workspace.resolve() not in self.roots:
            self.roots.append(self.workspace.resolve())
        self.blocked_ext = {e.lower() for e in sec.blocked_extensions}
        # дополнительно: не запускаем через ассоциации файлов то, что исполняется интерпретатором
        self.blocked_open_ext = self.blocked_ext | {".py", ".pyw", ".pyz", ".sh", ".vbs"}
        self.max_file_bytes = int(sec.max_file_bytes)

    def check(self, p: Path) -> Path:
        p = p.resolve(strict=False)
        for root in self.roots:
            if p == root or root in p.parents:
                return p
        raise ToolError("этот путь вне разрешённых папок (список в config.yaml → security.allowed_roots)")

    def resolve(self, raw: str) -> Path:
        s = (raw or "").strip().strip("\"'«»")
        if not s:
            raise ToolError("пустой путь")
        s = s.replace("\\", "/")
        first, _, rest = s.partition("/")
        key = _RU.get(first.strip().lower(), first.strip().lower())
        if key in self.known:
            base = self.known[key]
            p = base / rest if rest else base
        elif re.match(r"^[A-Za-z]:/", s) or s.startswith("/") or s.startswith("~"):
            p = Path(s).expanduser()
        else:
            p = self.workspace / s
        return self.check(p)

    def check_ext(self, p: Path, for_open: bool = False) -> None:
        ext = p.suffix.lower()
        blocked = self.blocked_open_ext if for_open else self.blocked_ext
        if ext in blocked:
            raise ToolError(f"файлы типа «{ext}» Кику не создаёт и не открывает из соображений безопасности")
