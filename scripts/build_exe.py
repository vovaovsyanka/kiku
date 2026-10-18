"""Собирает Kiku.exe (лаунчер без консоли). Работает в любой оболочке (cmd, PowerShell, Git Bash):

    python scripts/build_exe.py

Запускать из активированного окружения проекта. Результат: Kiku.exe в корне проекта.
Лаунчер не содержит моделей и библиотек: он находит окружение проекта и запускает `pythonw -m kiku gui`.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd) -> None:
    print(">", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def main() -> int:
    try:
        run([sys.executable, "-m", "pip", "install", "--quiet", "pyinstaller"])
        run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed", "--name", "Kiku",
             "--icon", ROOT / "kiku" / "ui" / "assets" / "kiku.ico", ROOT / "launcher.pyw"])
    except subprocess.CalledProcessError as e:
        print(f"\nСборка не удалась (код {e.returncode}). Пришлите вывод выше.")
        return 1
    shutil.copy2(ROOT / "dist" / "Kiku.exe", ROOT / "Kiku.exe")
    print("\nГотово: Kiku.exe лежит в корне проекта. Ярлык на рабочий стол: ПКМ — Отправить — Рабочий стол.")
    print("Если антивирус ругается на exe от PyInstaller — добавьте его в исключения: код лаунчера — launcher.pyw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
