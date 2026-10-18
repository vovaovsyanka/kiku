"""Придумывает голос Кику (один раз) и сохраняет его как референс для клонирования.

Модель Qwen3-TTS VoiceDesign (1.7B) создаёт голос по текстовому описанию.
Результат — voices/kiku_ref.wav + voices/kiku_ref.txt; дальше быстрая модель 0.6B-Base
клонирует этот голос для всех реплик. На 4 ГБ видеопамяти 1.7B не помещается,
поэтому по умолчанию запуск идёт на CPU (одна короткая фраза — до нескольких минут).

    python scripts/design_voice.py
    python scripts/design_voice.py --instruct "Calm mature female voice, low pitch, warm and polite"
    python scripts/design_voice.py --device cuda:0 --dtype bfloat16     # если видеопамяти хватает

Своё вместо дизайна: положите свою запись (3–10 секунд, чистая речь) в voices/kiku_ref.wav,
а дословный текст записи — в voices/kiku_ref.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kiku.env import setup_environment  # noqa: E402

setup_environment(offline=False)

from kiku.config import load_config, resolve_path  # noqa: E402

DEFAULT_INSTRUCT = ("Young adult female voice, warm, polite and gentle, slightly cheerful, "
                    "clear articulation, calm natural pace, like a kind and attentive housemaid.")
DEFAULT_TEXT = ("Здравствуйте, хозяин! Меня зовут Кику, и я буду заботиться о порядке "
                "на вашем компьютере. Чем могу помочь?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default=DEFAULT_INSTRUCT, help="описание голоса (лучше на английском)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="фраза, которую произнесёт голос (на русском)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"Загружаю {cfg.tts.qwen.design_repo} на {args.device}…")
    model = Qwen3TTSModel.from_pretrained(
        cfg.tts.qwen.design_repo, device_map=args.device, dtype=getattr(torch, args.dtype))
    print("Генерирую референсный голос…")
    with torch.inference_mode():
        wavs, sr = model.generate_voice_design(text=args.text, language=cfg.tts.language, instruct=args.instruct)

    wav_path = resolve_path(cfg.tts.qwen.ref_audio)
    txt_path = resolve_path(cfg.tts.qwen.ref_text_file)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), wavs[0], sr)
    txt_path.write_text(args.text, encoding="utf-8")
    print(f"Готово:\n  {wav_path}\n  {txt_path}\nПрослушайте файл; если голос не нравится — запустите с другим --instruct.")


if __name__ == "__main__":
    main()
