"""Запуск Кику без консоли.

Двойной клик по этому файлу (.pyw) или по Kiku.exe (его собирает scripts/build_exe.py из этого же файла).
Лаунчер только находит папку проекта и её виртуальное окружение и запускает там `pythonw -m kiku gui`.
Тяжёлое (torch, модели, библиотеки) остаётся в проекте, поэтому exe маленький и собирается за секунды.
Если окружение лежит в нестандартном месте — впишите путь к pythonw.exe первой строкой в launcher.txt.
Использует только стандартную библиотеку.
"""
import subprocess
import sys
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000


def alert(message: str) -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, message, "Кику", 0x10)
    except Exception:  # noqa: BLE001
        print(message, file=sys.stderr)


def find_project() -> Path | None:
    here = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    for d in [here, *here.parents]:
        if (d / "kiku" / "__main__.py").exists():
            return d
    return None


def find_pythonw(root: Path) -> Path | None:
    custom = root / "launcher.txt"
    if custom.exists():
        lines = custom.read_text(encoding="utf-8").strip().splitlines()
        if lines and Path(lines[0].strip().strip('"')).exists():
            return Path(lines[0].strip().strip('"'))
    for name in ("kikuenv", ".venv", "venv", "env"):
        candidate = root / name / "Scripts" / "pythonw.exe"
        if candidate.exists():
            return candidate
    found = sorted(root.glob("*/Scripts/pythonw.exe"))
    return found[0] if found else None


def main() -> int:
    root = find_project()
    if root is None:
        alert("Не нашла папку проекта Кику (рядом должна лежать папка kiku).\n"
              "Положите Kiku.exe в корень проекта.")
        return 1
    pythonw = find_pythonw(root)
    if pythonw is None:
        alert(f"Не нашла виртуальное окружение в {root}.\n"
              "Впишите путь к pythonw.exe первой строкой в файл launcher.txt.")
        return 1
    try:
        subprocess.Popen([str(pythonw), "-m", "kiku", "gui"], cwd=str(root), creationflags=CREATE_NO_WINDOW,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        alert(f"Не удалось запустить Кику: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
