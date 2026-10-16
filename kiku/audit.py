"""Журнал действий: каждый вызов инструмента пишется в data/audit.jsonl"""
import json
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **event) -> None:
        event["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
