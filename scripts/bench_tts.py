"""Замер скорости Qwen3-TTS на этой видеокарте при разных настройках (RTF < 1 — быстрее реального времени).

    python scripts/bench_tts.py
Перебирает: bfloat16+sdpa, float16+sdpa, bfloat16 без явной реализации внимания. Результат — таблица.
"""
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kiku.env import setup_environment  # noqa: E402

setup_environment(offline=True)

from kiku.config import load_config, resolve_path  # noqa: E402

TEXTS = ["Открываю рабочий стол.", "Хорошо, хозяин, сейчас всё сделаю.",
         "Файл заметки создан на рабочем столе, можете его открыть."]


def main() -> None:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    transformers.logging.set_verbosity_error()
    cfg = load_config().tts
    path = snapshot_download(repo_id=cfg.base_repo, local_files_only=True)
    ref_audio, ref_text = resolve_path(cfg.ref_audio), resolve_path(cfg.ref_text_file).read_text(encoding="utf-8").strip()
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'нет'}\n")

    results = []
    for dtype_name, attn in (("bfloat16", "sdpa"), ("float16", "sdpa"), ("bfloat16", None)):
        label = f"{dtype_name} / attn={attn}"
        try:
            kwargs = dict(device_map=cfg.device, dtype=getattr(torch, dtype_name))
            if attn:
                kwargs["attn_implementation"] = attn
            model = Qwen3TTSModel.from_pretrained(path, **kwargs)
            prompt = model.create_voice_clone_prompt(ref_audio=str(ref_audio), ref_text=ref_text)
            rtfs = []
            for i, text in enumerate(TEXTS * 2):          # первый проход — прогрев
                t0 = time.time()
                with torch.inference_mode():
                    wavs, sr = model.generate_voice_clone(text=text, language=cfg.language, voice_clone_prompt=prompt)
                took, dur = time.time() - t0, len(wavs[0]) / sr
                if i >= len(TEXTS):
                    rtfs.append(took / dur)
                    print(f"  {label}: {dur:.1f} с звука за {took:.1f} с (RTF {took / dur:.2f})")
            results.append((label, sum(rtfs) / len(rtfs)))
            del model, prompt
        except Exception as e:  # noqa: BLE001
            results.append((label, f"ошибка: {type(e).__name__}: {e}"))
        gc.collect()
        torch.cuda.empty_cache()

    print("\nИтог (RTF, меньше — лучше):")
    for label, v in results:
        print(f"  {label:32s} {v if isinstance(v, str) else f'{v:.2f}'}")


if __name__ == "__main__":
    main()
