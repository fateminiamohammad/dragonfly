"""One model thread runs every forward pass. Requests queue their records; the thread takes everything queued when it
becomes free (up to max_batch) and runs it as one batch. Under load this batches naturally; when idle a request runs
immediately, with no fixed wait window.

Two priorities: online requests (0) always go first; batch jobs (1, /v1/batch) fill whatever room is left in a forward
pass, so a million-row map-reduce never delays live traffic.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from concurrent.futures import Future

from .engine import Engine


class Worker:
    def __init__(self, engine: Engine, max_batch: int = 64):
        self.engine = engine
        self.max_batch = max_batch
        self.queue: queue.PriorityQueue = queue.PriorityQueue()  # (priority, sequence, (record, future) | None)
        self._seq = itertools.count()
        self.stopping = threading.Event()
        self.batches = 0
        self.requests = 0
        self.thread = threading.Thread(target=self._run, name="dragonfly-model", daemon=True)
        self.thread.start()

    def submit(self, record: dict, low_priority: bool = False) -> Future:
        """-> Future of (probabilities per question, {"tokens", "tiers", "latency_ms", "batch_size"})."""
        if self.stopping.is_set():
            raise RuntimeError("the model worker is stopping")
        done: Future = Future()
        self.queue.put((1 if low_priority else 0, next(self._seq), (record, done)))
        return done

    def close(self) -> None:
        self.stopping.set()
        self.queue.put((-1, next(self._seq), None))  # wake the thread
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stopping.is_set():
            item = self.queue.get()[2]
            if item is None:
                continue
            batch = [item]
            while len(batch) < self.max_batch:
                try:
                    nxt = self.queue.get_nowait()[2]
                except queue.Empty:
                    break
                if nxt is not None:
                    batch.append(nxt)
            try:
                self._answer(batch)
            except Exception:
                # one bad record (too long, empty option) must not fail its neighbours: run them one at a time
                for item in batch:
                    try:
                        self._answer([item])
                    except Exception as e:
                        item[1].set_exception(e)

    def _answer(self, batch: list[tuple[dict, Future]]) -> None:
        started = time.perf_counter()
        probs, tokens, tiers = self.engine.probs([rec for rec, _ in batch])
        ms = (time.perf_counter() - started) * 1000
        self.batches += 1
        self.requests += len(batch)
        for (_, fut), p, n, t in zip(batch, probs, tokens, tiers):
            fut.set_result((p, {"tokens": n, "tiers": t, "latency_ms": round(ms, 2), "batch_size": len(batch)}))
