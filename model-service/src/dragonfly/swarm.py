"""The swarm: many specialist dragonflies behind one endpoint, sharing one GPU.

A specialist is a small checkpoint trained for one job, kept in `<root>/<name>/` with a `specialist.json` card:

    {"name": "invoices", "description": "Invoices and payment documents", "tier": "S" | "M", "metrics": {...}}

  - tier S: a full Dragonfly-S checkpoint (~0.6 GB on disk, ~0.3 GB VRAM). Loaded on first use; the least recently used
    are unloaded beyond `max_loaded`.
  - tier M: a LoRA adapter + pointer head (~140 MB) on the SAME base as the general tier M. All adapters share one
    backbone in VRAM: switching copies the adapter's weights into the served LoRA tensors in place (`copy_`), so the
    CUDA graphs captured for those tensors stay valid and a switch costs a few milliseconds, not a model load.

A request picks a specialist with its `model` field. `"model": "auto"` asks the general tier S to choose among the
specialists' descriptions (one fast forward pass) and falls back to the general model when it is unsure.
Everything here runs on the single model thread (batching.Worker), so no locking is needed.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import torch

from .engine import Cascade, CheckpointConfig, Engine
from .models.decoder import LoRALinear, StateCache
from .schema import question_confidence

log = logging.getLogger("dragonfly.swarm")

GENERAL = ("dragonfly-latest", "jev-latest", "kev-latest", "general", "")
AUTO = "auto"


def read_card(folder: Path) -> dict | None:
    """The specialist.json of a folder (name defaults to the folder name), or None if it is not a specialist."""
    card_path = folder / "specialist.json"
    if not card_path.is_file() or not (folder / "config.json").is_file():
        return None
    card = json.loads(card_path.read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    card.setdefault("name", folder.name)
    card.setdefault("description", card["name"])
    card["tier"] = config["tier"]
    card["backbone"] = config["backbone"]
    card["path"] = str(folder)
    return card


def lora_modules(model: torch.nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


class AdapterBank:
    """Tier M adapters that take turns on one unmerged backbone. The general adapter is the host's own weights."""

    def __init__(self, host: Engine, general: str = "general"):
        if host.config.tier != "M":
            raise ValueError("adapters need a tier M host")
        self.host = host
        self.layers = lora_modules(host.model.backbone)
        if not self.layers:
            raise ValueError("the tier M host is merged: load it with merge=False to switch adapters")
        self.adapters: dict[str, tuple[dict, dict, float]] = {}
        self.active = general
        self.adapters[general] = self._snapshot()
        self.switches = 0
        self.caches: dict[str, StateCache] = {}

    def _snapshot(self) -> tuple[dict, dict, float]:
        lora = {f"{n}.{w}": getattr(m, w).weight.detach().clone() for n, m in self.layers.items() for w in ("lora_a", "lora_b")}
        head = {k: v.detach().clone() for k, v in self.host.model.head.state_dict().items()}
        return lora, head, self.host.config.temperature

    def add(self, name: str, path: str) -> None:
        root = Path(path)
        config = CheckpointConfig(**json.loads((root / "config.json").read_text()))
        host = self.host.config
        if (config.backbone, config.lora_r, config.head_dim, config.head_norm) != \
                (host.backbone, host.lora_r, host.head_dim, host.head_norm):
            raise ValueError(f"adapter {name} was trained for {config.backbone} r={config.lora_r} "
                             f"head_dim={config.head_dim} head_norm={config.head_norm}; the host serves "
                             f"{host.backbone} r={host.lora_r} head_dim={host.head_dim} head_norm={host.head_norm}")
        dev = self.host.device
        saved = torch.load(root / "lora.pt", map_location=dev, weights_only=True)
        lora = {}
        for n, m in self.layers.items():
            for w in ("lora_a", "lora_b"):
                key = f"{n}.{w}.weight"
                if key not in saved:
                    raise ValueError(f"adapter {name} has no {key}")
                lora[f"{n}.{w}"] = saved[key].to(getattr(m, w).weight.dtype)
        head = torch.load(root / "head.pt", map_location=dev, weights_only=True)
        self.adapters[name] = (lora, head, config.temperature)

    @torch.no_grad()
    def activate(self, name: str) -> None:
        if name == self.active:
            return
        lora, head, temperature = self.adapters[name]
        dst = [getattr(m, w).weight for m in self.layers.values() for w in ("lora_a", "lora_b")]
        src = [lora[f"{n}.{w}"] for n in self.layers for w in ("lora_a", "lora_b")]
        head_state = self.host.model.head.state_dict()
        dst += list(head_state.values())
        src += [head[k] for k in head_state]
        torch._foreach_copy_(dst, src)  # one fused launch group instead of ~500 small copies
        self.host.config.temperature = temperature
        if self.host.state_cache is not None:
            # cached KV depends on the adapter: each adapter keeps its own cache
            self.caches[self.active] = self.host.state_cache
            self.host.state_cache = self.caches.get(name) or StateCache(self.host.state_cache.size,
                                                                        self.host.state_cache.max_tokens)
        self.active = name
        self.switches += 1


class Swarm:
    """Engine-compatible (probs / describe / config / device): routes each record to its specialist."""

    def __init__(self, general, root: str | None, max_loaded: int = 4, route_threshold: float = 0.5,
                 load_engine=None, bank: AdapterBank | None = None, prepare=None):
        """load_engine(path) -> Engine. With `prepare`, load_engine runs on a background thread (so it must not
        overlap a CUDA-graph capture) and prepare(engine) -> Engine finishes it on the model thread."""
        self.general = general  # an Engine or a Cascade
        self.root = Path(root) if root else None
        self.max_loaded = max_loaded
        self.route_threshold = route_threshold
        self.load_engine = load_engine or (lambda path: Engine.load(path, general.device))
        self.bank = bank
        self.prepare = prepare
        self._lock = threading.Lock()
        self._ready: dict[str, tuple[str, Engine]] = {}  # preloaded on the CPU, waiting for first use
        self._loading: set[str] = set()
        self.config = general.config
        self.device = general.device
        self.cards: dict[str, dict] = {}
        self.loaded: OrderedDict[str, Engine] = OrderedDict()  # tier S specialists, LRU order
        self.served: dict[str, int] = {}
        self._reload = threading.Event()
        self.reload()

    # ---- registry ------------------------------------------------------------------------------------------------
    def request_reload(self) -> None:
        """Thread-safe: the model thread rescans the registry before its next batch. New or retrained tier S
        specialists start loading from disk right away, so their first request doesn't wait for it."""
        self._reload.set()
        self._preload(self.scan())

    def scan(self) -> dict[str, dict]:
        """The specialist cards on disk (reads files only; safe from any thread)."""
        cards = {}
        if self.root is not None and self.root.is_dir():
            for folder in sorted(self.root.iterdir()):
                try:
                    card = read_card(folder)
                except (OSError, ValueError, KeyError) as e:
                    log.warning("skipping specialist %s: %s", folder.name, e)
                    continue
                if card is None or card["name"] in GENERAL or card["name"] == AUTO:
                    continue
                if card["tier"] == "M" and self.bank is None:
                    continue
                cards[card["name"]] = card
        return cards

    def reload(self) -> None:
        """Model thread only: applies the registry on disk (loads tier M adapters, drops removed specialists)."""
        self._reload.clear()
        cards = {}
        if self.root is not None and self.root.is_dir():
            for folder in sorted(self.root.iterdir()):
                try:
                    card = read_card(folder)
                except (OSError, ValueError, KeyError) as e:
                    log.warning("skipping specialist %s: %s", folder.name, e)
                    continue
                if card is None:
                    continue
                if card["name"] in GENERAL or card["name"] == AUTO:
                    log.warning("skipping specialist %s: reserved name", card["name"])
                    continue
                if card["tier"] == "M":
                    if self.bank is None:
                        log.warning("skipping tier M specialist %s: no tier M host (set DRAGONFLY_CHECKPOINT_M)",
                                    card["name"])
                        continue
                    try:
                        self.bank.add(card["name"], card["path"])
                    except (OSError, ValueError) as e:
                        log.warning("skipping specialist %s: %s", card["name"], e)
                        continue
                cards[card["name"]] = card
        for gone in set(self.loaded) - set(cards):
            del self.loaded[gone]
        for name in set(self.cards) & set(cards):
            version = self.version_of(cards[name])
            if self.version_of(self.cards[name]) != version:  # retrained in place
                self.loaded.pop(name, None)
                with self._lock:
                    if name in self._ready and self._ready[name][0] != version:
                        del self._ready[name]
        if self.bank is not None:
            for gone in set(self.bank.adapters) - set(cards) - {"general"}:
                if self.bank.active == gone:
                    self.bank.activate("general")
                del self.bank.adapters[gone]
                self.bank.caches.pop(gone, None)
        with self._lock:
            for gone in set(self._ready) - set(cards):
                del self._ready[gone]
        self.cards = cards
        log.info("swarm: %d specialists (%s)", len(cards), ", ".join(cards) or "none")
        self._preload(cards)

    def _preload(self, cards: dict[str, dict]) -> None:
        """Load new tier S specialists on a background thread, so a first request doesn't wait for the disk or the
        host-to-GPU copy. load_engine must not overlap a CUDA-graph capture (serve.load_specialist holds
        models.graphs.CAPTURE_LOCK while copying); prepare() then runs on the model thread."""
        if self.prepare is None:
            return
        def current(name: str, card: dict) -> bool:
            """Already loaded (same version) or loading."""
            version = self.version_of(card)
            applied = self.cards.get(name)
            if name in self.loaded and applied is not None and self.version_of(applied) == version:
                return True
            return name in self._loading or (name in self._ready and self._ready[name][0] == version)

        with self._lock:
            todo = [(n, c) for n, c in cards.items() if c["tier"] == "S" and not current(n, c)]
            room = max(0, self.max_loaded - len(self._ready) - len(self._loading))
            todo = todo[:room]
            self._loading.update(n for n, _ in todo)
        if not todo:
            return

        def run():
            for name, card in todo:
                try:
                    engine = self.load_engine(card["path"])
                    with self._lock:
                        self._ready[name] = (self.version_of(card), engine)
                except Exception as e:
                    log.warning("preloading specialist %s failed: %s", name, e)
                finally:
                    with self._lock:
                        self._loading.discard(name)
        threading.Thread(target=run, name="dragonfly-swarm-preload", daemon=True).start()

    @staticmethod
    def version_of(card: dict) -> str:
        return f"@{card.get('job') or card.get('created') or ''}"

    def version(self, name: str) -> str:
        """Changes when a specialist is retrained (cache keys include it)."""
        card = self.cards.get(name)
        return "" if card is None else self.version_of(card)

    def list(self) -> list[dict]:
        # a reload may still be pending on the model thread: show what is on disk now
        cards = self.scan() if self._reload.is_set() else self.cards
        return [{"name": c["name"], "description": c["description"], "tier": c["tier"],
                 "metrics": c.get("metrics", {}), "created": c.get("created"),
                 "loaded": c["tier"] == "M" or c["name"] in self.loaded,
                 "served": self.served.get(c["name"], 0)} for c in cards.values()]

    # ---- serving -------------------------------------------------------------------------------------------------
    def _engine(self, name: str):
        card = self.cards[name]
        if card["tier"] == "M":
            return _AdapterView(self.bank, name)
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]
        with self._lock:
            ready = self._ready.pop(name, None)
        if ready is not None and ready[0] == self.version_of(card):
            engine = ready[1]
        else:
            engine = self.load_engine(card["path"])
        if self.prepare is not None:
            engine = self.prepare(engine)  # CUDA graphs on (captured per shape on first use)
        self.loaded[name] = engine
        while len(self.loaded) > self.max_loaded:
            evicted, _ = self.loaded.popitem(last=False)
            log.info("swarm: unloaded %s (least recently used)", evicted)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return engine

    def _general(self):
        """The general model, with the general adapter active on a shared tier M host."""
        if self.bank is not None:
            self.bank.activate("general")
        return self.general

    def route(self, records: list[dict]) -> list[str]:
        """model "auto": the general tier S picks a specialist per record from their descriptions (one pass)."""
        names = list(self.cards)
        if not names:
            return ["general"] * len(records)
        options = [self.cards[n]["description"] for n in names] + ["none of these: a general question"]
        router = self.general.small if isinstance(self.general, Cascade) else self.general
        if self.bank is not None and router is self.bank.host:
            self.bank.activate("general")
        question = {"instr": "Which specialist should handle this document?", "options": options, "qtype": "choice"}
        probs, _, _ = router.probs([{"state": r["state"], "questions": [question]} for r in records])
        out = []
        for p in probs:
            best = max(range(len(options)), key=p[0].__getitem__)
            sure = question_confidence("choice", p[0]) >= self.route_threshold
            out.append(names[best] if sure and best < len(names) else "general")
        return out

    def probs(self, records: list[dict]):
        if self._reload.is_set():
            self.reload()
        wanted = [r.get("specialist") or "general" for r in records]
        auto = [i for i, w in enumerate(wanted) if w == AUTO]
        if auto:
            for i, name in zip(auto, self.route([records[i] for i in auto])):
                wanted[i] = name
        groups: dict[str, list[int]] = {}
        for i, w in enumerate(wanted):
            if w in GENERAL:
                w = "general"
            elif w not in self.cards:
                raise ValueError(f"unknown model {w!r}: see GET /v1/specialists")
            groups.setdefault(w, []).append(i)
        probs: list = [None] * len(records)
        tokens = [0] * len(records)
        tiers: list = [None] * len(records)
        for name, idx in groups.items():
            engine = self._general() if name == "general" else self._engine(name)
            p, t, tr = engine.probs([records[i] for i in idx])
            for i, pi, ti, tri in zip(idx, p, t, tr):
                probs[i], tokens[i], tiers[i] = pi, ti, tri
                records[i]["routed_to"] = name
            self.served[name] = self.served.get(name, 0) + len(idx)
        return probs, tokens, tiers

    def describe(self) -> dict:
        out = dict(self.general.describe())
        out["specialists"] = len(self.cards)
        if self.bank is not None:
            out["adapter_switches"] = self.bank.switches
        return out


class _AdapterView:
    """A tier M specialist as an engine: activates its adapter, then runs the shared host."""

    def __init__(self, bank: AdapterBank, name: str):
        self.bank, self.name = bank, name

    def probs(self, records):
        self.bank.activate(self.name)
        return self.bank.host.probs(records)
