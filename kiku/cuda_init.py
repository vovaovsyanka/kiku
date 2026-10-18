"""Порядок инициализации CUDA на Windows.

Ошибка «Could not load symbol cudnnGetLibConfig. Error code 127» появляется, когда в одном процессе
встречаются две разные версии cuDNN: одну приносит torch (папка torch/lib), другую — ctranslate2/faster-whisper
(из PATH или пакета nvidia-cudnn). Windows загружает DLL по имени: кто первым загрузил cudnn64_9.dll, тот и «побеждает».
Решение: ВСЕГДА первым инициализировать torch (и его cuDNN), а уже потом создавать модель Whisper.
"""
import logging

log = logging.getLogger("kiku.cuda")


def init_cuda_first() -> str:
    """Импортирует torch и принудительно загружает его cuDNN малой свёрткой. Возвращает описание для лога."""
    try:
        import torch
    except ImportError:
        return "torch не установлен"
    try:
        if not torch.cuda.is_available():
            return "CUDA недоступна (torch без CUDA или нет драйвера)"
        x = torch.zeros(1, 3, 16, 16, device="cuda")
        w = torch.zeros(8, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)          # заставляет torch загрузить свою cuDNN
        torch.cuda.synchronize()
        return f"{torch.cuda.get_device_name(0)}, cuDNN {torch.backends.cudnn.version()}"
    except Exception as e:  # noqa: BLE001
        log.warning("Инициализация CUDA не удалась: %s", e)
        return f"ошибка инициализации CUDA: {e}"
