"""Интерфейс языковой модели. Чтобы подключить другой движок — реализуйте chat_json()."""
from typing import Protocol


class LLMBackend(Protocol):
    def chat_json(self, messages: list[dict], schema: dict | None = None) -> str:
        """Вернуть ответ модели — строку с JSON-объектом."""
        ...


def build_llm(cfg) -> LLMBackend:
    if cfg.llm.backend == "llama_cpp":
        from .llama_cpp_backend import LlamaCppBackend
        return LlamaCppBackend(cfg.llm)
    raise ValueError(f"Неизвестный backend LLM: {cfg.llm.backend}")
