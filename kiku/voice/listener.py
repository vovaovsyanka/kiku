"""Микрофон + детектор голосовой активности (Silero VAD). Возвращает одну законченную фразу."""
import collections
import queue
import time

import numpy as np


class MicListener:
    def __init__(self, audio_cfg):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        self._torch = torch
        self._VADIterator = VADIterator
        self.model = load_silero_vad()
        self.cfg = audio_cfg
        self.sr = int(audio_cfg.sample_rate)
        if self.sr != 16000:
            raise ValueError("Silero VAD в этой сборке настроен на 16000 Гц")
        self.chunk = 512  # размер кадра для Silero VAD при 16 кГц

    def listen(self, timeout: float | None = None, stop=None, on_speech_start=None) -> np.ndarray | None:
        """Ждёт речь и возвращает фразу (float32, 16 кГц).
        None — если вышел timeout или stop() вернул True. on_speech_start() вызывается в момент начала речи."""
        import sounddevice as sd

        q: queue.Queue = queue.Queue()

        def callback(indata, frames, t, status):  # вызывается в потоке PortAudio
            q.put(indata[:, 0].copy())

        vad = self._VADIterator(
            self.model, threshold=float(self.cfg.vad_threshold), sampling_rate=self.sr,
            min_silence_duration_ms=int(self.cfg.min_silence_ms), speech_pad_ms=150)
        vad.reset_states()
        preroll = collections.deque(maxlen=10)   # ~320 мс до начала речи, чтобы не резать первое слово
        buf: list = []
        recording = False
        started = time.time()
        max_chunks = int(float(self.cfg.max_utterance_sec) * self.sr / self.chunk)

        with sd.InputStream(samplerate=self.sr, channels=1, dtype="float32", blocksize=self.chunk,
                            device=self.cfg.input_device, callback=callback):
            while True:
                if stop is not None and not recording and stop():
                    return None
                try:
                    chunk = q.get(timeout=0.3)
                except queue.Empty:
                    chunk = None
                if chunk is None:
                    if timeout and not recording and time.time() - started > timeout:
                        return None
                    continue
                event = vad(self._torch.from_numpy(chunk))
                if not recording:
                    preroll.append(chunk)
                    if event and "start" in event:
                        recording = True
                        buf = list(preroll)
                        if on_speech_start:
                            on_speech_start()
                    elif timeout and time.time() - started > timeout:
                        return None
                else:
                    buf.append(chunk)
                    if (event and "end" in event) or len(buf) >= max_chunks:
                        return np.concatenate(buf)
