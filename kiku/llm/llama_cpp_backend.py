"""Запуск GGUF-модели прямо в процессе приложения (библиотека llama-cpp-python).

Никаких внешних серверов (Ollama, LM Studio) не нужно: модель грузится из ./models/hf.
Ответ ограничивается JSON-грамматикой, поэтому даже небольшая модель
всегда возвращает синтаксически корректный JSON с допустимым именем инструмента.
"""
import glob
import logging
import os
import time

log = logging.getLogger("kiku.llm")


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


class LlamaCppBackend:
    def __init__(self, llm_cfg):
        from llama_cpp import Llama

        self.cfg = llm_cfg
        path = find_gguf(llm_cfg.repo_id, llm_cfg.file_pattern)
        log.info("Загружаю LLM: %s", path)
        self.llm = Llama(
            model_path=path,
            n_ctx=int(llm_cfg.n_ctx),
            n_gpu_layers=int(llm_cfg.n_gpu_layers),
            n_threads=(int(llm_cfg.n_threads) or None),
            verbose=False,
        )

    def device_info(self) -> str:
        import llama_cpp
        n = int(self.cfg.n_gpu_layers)
        if n > 0 and llama_cpp.llama_supports_gpu_offload():
            return f"GPU: {n} слоёв выгружено, остальное на CPU"
        why = "сборка llama-cpp-python без CUDA" if not llama_cpp.llama_supports_gpu_offload() else "n_gpu_layers: 0"
        return f"CPU ({why})"

    def device_label(self) -> str:
        import llama_cpp
        return "GPU" if int(self.cfg.n_gpu_layers) > 0 and llama_cpp.llama_supports_gpu_offload() else "CPU"

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
