"""Запуск GGUF-модели прямо в процессе приложения (библиотека llama-cpp-python).

Никаких внешних серверов (Ollama, LM Studio) не нужно: модель грузится из ./models/hf.
Ответ ограничивается JSON-грамматикой, поэтому даже небольшая модель
всегда возвращает синтаксически корректный JSON с допустимым именем инструмента.

Видеокарта. Модель-MoE (Qwen3-30B-A3B: 30 млрд параметров, но на каждый токен работают только 3 млрд) целиком в 4 ГБ
не помещается. Зато её «эксперты» (ffn_*_exps — почти весь вес) можно оставить в оперативной памяти, а внимание, KV-кэш
и маленькие слои держать на GPU: так делает `--n-cpu-moe` в llama.cpp. llama-cpp-python этот параметр не пробрасывает,
поэтому здесь он выставляется вручную (см. cpu_experts_override).
"""
import contextlib
import ctypes
import glob
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("kiku.llm")

GGML_TYPE_Q8_0 = 8


def find_gguf(repo_id: str, pattern: str) -> str:
    """Находит скачанный GGUF-файл в локальном кэше Hugging Face (без сети)."""
    from huggingface_hub import snapshot_download
    try:
        folder = snapshot_download(repo_id=repo_id, allow_patterns=[pattern], local_files_only=True)
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError(
            f"Модель {repo_id} не найдена локально. Скачайте её: python scripts/download_models.py --only llm"
        ) from e
    files = sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"В {folder} нет файлов по маске {pattern}. Запустите: python scripts/download_models.py --only llm")
    return files[0]  # при разбитых на части GGUF первая часть содержит ссылку на остальные


class _BuftOverride(ctypes.Structure):          # struct llama_model_tensor_buft_override
    _fields_ = [("pattern", ctypes.c_char_p), ("buft", ctypes.c_void_p)]


@contextlib.contextmanager
def cpu_experts_override(n_layers: int):
    """На время загрузки модели: веса экспертов первых n_layers слоёв остаются в оперативной памяти (аналог --n-cpu-moe)."""
    import llama_cpp.llama_cpp as lc

    if n_layers <= 0:
        yield
        return
    cpu_lib = ctypes.CDLL(str(Path(lc.__file__).parent / "lib" / "ggml-base.dll"))
    cpu_lib.ggml_backend_cpu_buffer_type.restype = ctypes.c_void_p
    layers = "|".join(str(i) for i in range(n_layers))
    overrides = (_BuftOverride * 2)(                                   # массив заканчивается пустой записью
        _BuftOverride(rf"blk\.(?:{layers})\.ffn_(?:up|down|gate)_exps\.".encode(), cpu_lib.ggml_backend_cpu_buffer_type()),
        _BuftOverride(None, None))
    original = lc.llama_model_default_params

    def patched():
        params = original()
        params.tensor_buft_overrides = ctypes.cast(overrides, ctypes.c_void_p)
        return params

    lc.llama_model_default_params = patched
    try:
        yield
    finally:
        lc.llama_model_default_params = original


def load_llama(path: str, n_ctx: int, n_gpu_layers: int, n_threads: int | None, n_cpu_moe: int = 0, kv_q8: bool = False):
    """Создаёт Llama. n_cpu_moe > 0 — эксперты стольких первых слоёв остаются на CPU. kv_q8 — KV-кэш в 8 бит (вдвое меньше VRAM)."""
    from llama_cpp import Llama

    extra = dict(flash_attn=True, type_k=GGML_TYPE_Q8_0, type_v=GGML_TYPE_Q8_0) if kv_q8 else {}
    with cpu_experts_override(n_cpu_moe if n_gpu_layers != 0 else 0):
        return Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads, verbose=False, **extra)


class LlamaCppBackend:
    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        path = find_gguf(llm_cfg.repo_id, llm_cfg.file_pattern)
        log.info("Загружаю LLM: %s", path)
        t0 = time.time()
        self.llm = load_llama(
            path, int(llm_cfg.n_ctx), int(llm_cfg.n_gpu_layers), int(llm_cfg.n_threads) or None,
            n_cpu_moe=int(getattr(llm_cfg, "n_cpu_moe", 0)), kv_q8=bool(getattr(llm_cfg, "kv_cache_q8", False)))
        log.info("LLM загружена за %.1f с", time.time() - t0)

    def _on_gpu(self) -> bool:
        import llama_cpp
        return int(self.cfg.n_gpu_layers) != 0 and bool(llama_cpp.llama_supports_gpu_offload())

    def device_info(self) -> str:
        import llama_cpp
        if not llama_cpp.llama_supports_gpu_offload():
            return "CPU (сборка llama-cpp-python без CUDA)"
        n = int(self.cfg.n_gpu_layers)
        if n == 0:
            return "CPU (llm.n_gpu_layers: 0)"
        moe = int(getattr(self.cfg, "n_cpu_moe", 0))
        where = f"эксперты {moe} слоёв в ОЗУ" if moe else "все веса слоёв на GPU"
        return f"GPU: слоёв {'все' if n < 0 or n > 200 else n}, {where}"

    def device_label(self) -> str:
        return "GPU" if self._on_gpu() else "CPU"

    def warmup(self, messages) -> None:
        """Прогрев кэша: модель заранее «прочитывает» системный промпт (без генерации ответа)."""
        t0 = time.time()
        self.llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0)
        log.info("LLM прогрета за %.1f с", time.time() - t0)

    def _run(self, **kwargs) -> str:
        t0 = time.time()
        out = self.llm.create_chat_completion(**kwargs)
        took = time.time() - t0
        n = (out.get("usage") or {}).get("completion_tokens")
        if n:
            log.info("LLM: %d ток. за %.1f с (%.1f ток/с, вместе с обработкой промпта)", n, took, n / max(took, 1e-6))
        return out["choices"][0]["message"]["content"]

    def chat_json(self, messages, schema=None) -> str:
        kwargs = dict(
            messages=messages,
            temperature=float(self.cfg.temperature),
            max_tokens=int(self.cfg.max_tokens),
        )
        if schema:
            try:
                return self._run(response_format={"type": "json_object", "schema": schema}, **kwargs)
            except Exception as e:  # noqa: BLE001 — на случай, если схема не конвертировалась в грамматику
                log.warning("JSON-схема не сработала (%s), пробую простой JSON-режим", e)
        return self._run(response_format={"type": "json_object"}, **kwargs)
