"""Диагностика: что реально увидят torch, llama.cpp, ctranslate2 и аудио на этой машине.

    python scripts/check_env.py
"""
import sys

print(f"Python {sys.version.split()[0]}")

try:
    import torch
    print(f"\n[torch] {torch.__version__}, сборка CUDA: {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f"[torch] GPU: {p.name}, архитектура sm_{p.major}{p.minor}")
        print(f"[torch] VRAM: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ")
        print(f"[torch] bfloat16 аппаратно: {torch.cuda.is_bf16_supported()}  (False -> tts.dtype: float16)")
    else:
        print("[torch] CUDA НЕДОСТУПНА -> TTS пойдёт на CPU (медленно). Нужна CUDA-сборка torch, см. README, шаг 1.")
except ImportError:
    print("\n[torch] не установлен")

try:
    import llama_cpp
    print(f"\n[llama-cpp-python] {llama_cpp.__version__}, выгрузка слоёв на GPU: {llama_cpp.llama_supports_gpu_offload()}")
    if not llama_cpp.llama_supports_gpu_offload():
        print("[llama-cpp-python] сборка только под CPU. Для LLM с n_gpu_layers: 0 это нормально.")
except Exception as e:  # noqa: BLE001
    print(f"\n[llama-cpp-python] недоступен: {e}")

try:
    import ctranslate2
    print(f"\n[ctranslate2/faster-whisper] CUDA-устройств: {ctranslate2.get_cuda_device_count()}")
except Exception as e:  # noqa: BLE001
    print(f"\n[ctranslate2] недоступен: {e}")

try:
    import sounddevice as sd
    d = sd.query_devices()
    print(f"\n[audio] устройство ввода по умолчанию: {sd.default.device[0]}, вывода: {sd.default.device[1]}")
    for i, dev in enumerate(d):
        if dev["max_input_channels"] > 0:
            print(f"   вход #{i}: {dev['name']}")
except Exception as e:  # noqa: BLE001
    print(f"\n[audio] ошибка: {e}")
