"""Конвейер озвучки: синтез кусками + воспроизведение без разрывов.

Если синтезатор медленнее реального времени (RTF > 1), играть каждый кусок сразу, как он готов, нельзя — речь «рвётся»
паузами. Ждать весь текст — долго. Компромисс: играть начинаем, когда накоплено ровно столько звука, чтобы оставшаяся
часть успела досинтезироваться, пока играет начало. Из условия «буфер не должен опустеть» следует старт, когда готова
доля (RTF - 1) / RTF звука. Длина реплики оценивается по числу символов и поправляется по реально измеренным значениям.
Быстрый синтезатор (RTF <= 1) просто играет куски по мере готовности.

on_start(text) вызывается, когда реплика начинает звучать, on_end() — когда договорила. Не зависит ни от torch,
ни от звуковой карты: синтез и воспроизведение передаются функциями (поэтому легко тестируется).
"""
import logging
import queue
import threading
import time

import numpy as np

log = logging.getLogger("kiku.speech")

GAP_SEC = 0.08   # пауза между кусками


class _Utterance:
    def __init__(self, chunks):
        self.chunks = chunks
        self.ready: list = []        # [(wav, sr)]
        self.buffered = 0.0          # секунд готового звука
        self.done = False
        self.cv = threading.Condition()


def join_audio(batch) -> tuple[np.ndarray, int]:
    """Склеивает куски [(wav, sr)] в один массив, вставляя короткие паузы."""
    sr = batch[0][1]
    gap = np.zeros(int(GAP_SEC * sr), dtype=np.float32)
    parts = []
    for wav, _ in batch:
        parts += [wav, gap]
    return np.concatenate(parts[:-1]), sr


class SpeechPipeline:
    def __init__(self, synth_fn, play_fn, split_fn, chunk_chars=120, cache=None,
                 on_start=None, on_end=None, stop_fn=None, initial_rtf=2.5, safety=1.15):
        self.synth_fn, self.play_fn, self.split_fn = synth_fn, play_fn, split_fn
        self.chunk_chars, self.cache = int(chunk_chars), cache
        self.on_start, self.on_end, self.stop_fn = on_start, on_end, stop_fn
        self.rtf = float(initial_rtf)       # сколько секунд работы на секунду звука (скользящая оценка)
        self.spc = 0.07                     # секунд звука на символ (скользящая оценка)
        self.safety = float(safety)
        self._in: queue.Queue = queue.Queue()
        self._pending = 0
        self._epoch = 0
        self._announced = False             # on_start для текущей реплики уже вызван
        self._cv = threading.Condition()
        threading.Thread(target=self._worker, daemon=True, name="speech-worker").start()

    # ---------- публичное ----------
    def submit(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._cv:
            self._pending += 1
        self._in.put((self._epoch, text))

    def wait(self) -> None:
        with self._cv:
            while self._pending > 0:
                self._cv.wait(0.2)

    def idle(self) -> bool:
        return self._pending == 0

    def stop(self) -> None:
        self._epoch += 1
        try:
            while True:
                self._in.get_nowait()
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()
        except queue.Empty:
            pass
        if self.stop_fn:
            self.stop_fn()

    # ---------- внутреннее ----------
    def _announce(self, text: str) -> None:
        if not self._announced:
            self._announced = True
            if self.on_start:
                self.on_start(text)

    def _worker(self):
        while True:
            epoch, text = self._in.get()
            self._announced = False
            try:
                if epoch == self._epoch:
                    self._speak(epoch, text)
            except Exception:  # noqa: BLE001
                log.exception("Ошибка озвучки")
                if not self._announced:      # озвучка сломалась — текст всё равно должен дойти до хозяина
                    self._announce(text)
                    if self.on_end:
                        self.on_end()
            finally:
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()

    def _speak(self, epoch: int, text: str) -> None:
        cached = self.cache.get(text) if self.cache else None
        if cached is not None:                       # фраза уже синтезировалась раньше — играем сразу
            self._announce(text)
            try:
                self.play_fn(*cached)
            finally:
                if self.on_end:
                    self.on_end()
            return
        chunks = self.split_fn(text, self.chunk_chars) or [text]
        st = _Utterance(chunks)
        player = threading.Thread(target=self._player, args=(epoch, st, text), daemon=True, name="speech-player")
        player.start()
        try:
            for chunk in chunks:
                if epoch != self._epoch:
                    break
                t0 = time.time()
                wav, sr = self.synth_fn(chunk)
                took, dur = time.time() - t0, len(wav) / float(sr)
                if dur > 0:
                    self.rtf = 0.6 * self.rtf + 0.4 * (took / dur)
                    self.spc = 0.6 * self.spc + 0.4 * (dur / max(len(chunk), 1))
                with st.cv:
                    st.ready.append((wav, sr))
                    st.buffered += dur
                    st.cv.notify_all()
        finally:
            with st.cv:
                st.done = True
                st.cv.notify_all()
        player.join()
        if self.cache and epoch == self._epoch and st.ready:
            self.cache.put(text, *join_audio(st.ready))

    def _need_buffer(self, st: _Utterance) -> float:
        est_total = sum(len(c) for c in st.chunks) * self.spc
        if self.rtf <= 1.0:
            return 0.0
        return min(est_total, est_total * (1.0 - 1.0 / self.rtf) * self.safety)

    def _player(self, epoch: int, st: _Utterance, text: str) -> None:
        need = self._need_buffer(st)
        with st.cv:
            while not (st.done or (st.ready and st.buffered >= need) or epoch != self._epoch):
                st.cv.wait(0.1)
        if epoch != self._epoch or not st.ready:
            return
        self._announce(text)
        played = 0
        try:
            while epoch == self._epoch:
                with st.cv:
                    while played >= len(st.ready) and not st.done and epoch == self._epoch:
                        st.cv.wait(0.1)
                    batch = st.ready[played:]
                    played = len(st.ready)
                    finished = st.done and not batch
                if finished:
                    break
                if batch:
                    self.play_fn(*join_audio(batch))
        finally:
            if self.on_end:
                self.on_end()
