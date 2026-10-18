"""Отчёт «что на каком устройстве работает» — чтобы не гадать, где считаются модели."""


def report(asr=None, tts=None, llm=None) -> list[str]:
    lines = []
    for title, obj in (("Распознавание речи", asr), ("Озвучка", tts), ("Языковая модель", llm)):
        if obj is None:
            lines.append(f"{title}: не загружена")
            continue
        fn = getattr(obj, "device_info", None)
        try:
            lines.append(f"{title}: {fn() if fn else 'нет данных'}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"{title}: не удалось определить ({type(e).__name__})")
    return lines


def summary(asr=None, tts=None, llm=None) -> str:
    """Короткая строка для окна: «распознавание: cuda · озвучка: cuda:0 · LLM: CPU»."""
    parts = []
    for title, obj in (("распознавание", asr), ("озвучка", tts), ("LLM", llm)):
        fn = getattr(obj, "device_label", None) if obj is not None else None
        try:
            parts.append(f"{title}: {fn() if fn else 'нет'}")
        except Exception:  # noqa: BLE001
            parts.append(f"{title}: ?")
    return " · ".join(parts)
