@echo off
chcp 65001 >nul
cd /d "%~dp0"
rem Берём Python из виртуального окружения проекта (kikuenv, .venv, venv, env)
set PY=
for %%D in (kikuenv .venv venv env) do if exist "%%D\Scripts\python.exe" if not defined PY set PY=%%D\Scripts\python.exe
if not defined PY set PY=python
echo Использую: %PY%
%PY% -m pip install --quiet pyinstaller
if errorlevel 1 goto fail
%PY% -m PyInstaller --noconfirm --onefile --windowed --name Kiku --icon "kiku\ui\assets\kiku.ico" launcher.pyw
if errorlevel 1 goto fail
copy /y dist\Kiku.exe Kiku.exe >nul
echo.
echo Готово: Kiku.exe лежит в корне проекта. Запускайте его двойным щелчком (ярлык на рабочий стол: ПКМ - Отправить - Рабочий стол).
pause
exit /b 0
:fail
echo.
echo Сборка не удалась. Пришлите вывод выше.
pause
exit /b 1
