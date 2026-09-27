"""Answer cache. Decisions are deterministic (same state, same questions -> same probabilities), so a repeated request
is answered from memory without touching the GPU. Keyed by a hash of the record the model would see, i.e. after
plugins have rewritten the request."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


def record_key(record: dict, model: str) -> str:
    blob = json.dumps({"m": model, "s": record["state"], "q": [(q["instr"], q["options"]) for q in record["questions"]]},
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class AnswerCache:
    def __init__(self, size: int):
        self.size = size
        self.entries: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        if not self.size:
            return None
        with self.lock:
            value = self.entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value) -> None:
        if not self.size:
            return
        with self.lock:
            self.entries[key] = value
            self.entries.move_to_end(key)
            while len(self.entries) > self.size:
                self.entries.popitem(last=False)

    def stats(self) -> dict:
        return {"size": self.size, "entries": len(self.entries), "hits": self.hits, "misses": self.misses}
