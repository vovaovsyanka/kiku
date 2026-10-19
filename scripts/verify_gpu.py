"""Проверка, что нейросети РЕАЛЬНО работают на видеокарте, а не на процессоре.

    python scripts/verify_gpu.py

Каждая проверка идёт в отдельном процессе (если одна упадёт, остальные продолжатся) и печатает факты:
  torch   — есть ли CUDA, насколько быстрее матричные операции на GPU, чем на CPU;
  asr     — устройство Whisper (после torch, в том порядке, как в приложении) и загрузка GPU при распознавании;
  tts     — какой движок озвучки (tts.engine), на каком устройстве он считает и насколько быстро (RTF);
  llm     — собрана ли llama-cpp-python с поддержкой GPU и сколько слоёв на GPU сейчас.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def check_torch():
    from kiku.env import setup_environment
    setup_environment(offline=True)
    from kiku.cuda_init import init_cuda_first
    print("CUDA:", init_cuda_first())
    import torch
    if not torch.cuda.is_available():
        print("ИТОГ: CUDA недоступна — всё на CPU. Поставьте torch с CUDA (см. README, шаг 1).")
        raise SystemExit(2)
    free, total = torch.cuda.mem_get_info()
    print(f"Видеопамять: свободно {free / 2**30:.1f} из {total / 2**30:.1f} ГБ")
    n = 2048
    for dev, dtype in (("cpu", torch.float32), ("cuda", torch.bfloat16)):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        (a @ b)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            a @ b
        if dev == "cuda":
            torch.cuda.synchronize()
        print(f"Матричное умножение {n}x{n} ×5 на {dev}: {time.time() - t0:.3f} с")
    print("ИТОГ: torch видит GPU и считает на нём.")


def check_asr():
    from kiku.env import setup_environment
    setup_environment(offline=True)
    from kiku.cuda_init import init_cuda_first
    print("CUDA:", init_cuda_first())          # как в приложении: сначала torch/cuDNN
    import numpy as np
    from _gpu_tools import GpuSampler
    from kiku.config import load_config
    from kiku.voice.asr import WhisperASR
    asr = WhisperASR(load_config().asr)
    print("Whisper:", asr.device_info())
    audio = (np.random.randn(16000 * 6) * 0.01).astype(np.float32)
    with GpuSampler() as gpu:
        t0 = time.time()
        asr.transcribe(audio)
        print(f"Распознавание 6 с тишины/шума заняло {time.time() - t0:.1f} с")
    print(gpu.summary())
    print("ИТОГ:", "GPU" if "cuda" in asr.device_label() else "CPU", "— устройство Whisper:", asr.device_label())


def check_tts():
    from kiku.env import setup_environment
    setup_environment(offline=True)
    from kiku.cuda_init import init_cuda_first
    print("CUDA:", init_cuda_first())
    from _gpu_tools import GpuSampler
    from kiku.config import load_config
    from kiku.voice.tts import build_tts
    tts = build_tts(load_config().tts)
    print("Озвучка:", tts.device_info())
    tts.synth("Проверка.")                      # прогрев
    with GpuSampler() as gpu:
        t0 = time.time()
        wav, sr = tts.synth("Открываю рабочий стол, хозяин, и создаю файл заметки.")
        took = time.time() - t0
    print(f"{len(wav) / sr:.1f} с звука за {took:.1f} с (RTF {took / (len(wav) / sr):.2f})")
    print(gpu.summary())
    util = sum(gpu.util) / len(gpu.util) if gpu.util else None
    if util is not None and util < 50 and tts.device_label().startswith("cuda"):
        print("ВЫВОД: модель на GPU, но видеокарта загружена слабо. Скорость ограничена процессором и накладными расходами Python "
              "(тысячи мелких операций на каждую секунду звука), а не мощностью видеокарты. Проверьте, что ноутбук включён в розетку, "
              "план электропитания — «Высокая производительность», а режим графики — «NVIDIA»/«Высокая производительность».")
    print("ИТОГ: параметры на", tts.device_label())


def check_llm():
    from kiku.env import setup_environment
    setup_environment(offline=True)
    import llama_cpp
    from kiku.config import load_config
    n = int(load_config().llm.n_gpu_layers)
    sup = llama_cpp.llama_supports_gpu_offload()
    print(f"llama-cpp-python {llama_cpp.__version__}: поддержка GPU в сборке: {sup}; n_gpu_layers в config.yaml: {n}")
    if n != 0 and sup:
        moe = int(getattr(load_config().llm, "n_cpu_moe", 0))
        print(f"ИТОГ: LLM на GPU (слоёв: {'все' if n < 0 or n > 200 else n}" + (f", эксперты {moe} слоёв в ОЗУ)." if moe else ")."))
    else:
        print("ИТОГ: LLM работает на CPU" + ("" if sup else " (сборка llama-cpp-python без CUDA)") + ".")


CHECKS = {"torch": check_torch, "asr": check_asr, "tts": check_tts, "llm": check_llm}


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        CHECKS[sys.argv[2]]()
        return
    results = []
    for name in CHECKS:
        print(f"\n===== {name} =====", flush=True)
        code = subprocess.call([sys.executable, __file__, "--check", name])
        results.append((name, "ok" if code == 0 else f"ПРОБЛЕМА (код {code})"))
    print("\n===== сводка =====")
    for name, status in results:
        print(f"  {name:6s} {status}")


if __name__ == "__main__":
    main()
