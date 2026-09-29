"""The client. Questions are built from Python types; answers come back as Decision objects."""

from __future__ import annotations

import enum
import functools
import inspect
import json
import typing
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx


class DragonflyError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


@dataclass(frozen=True)
class Scale:
    """An ordered scale (a `score` question): levels from lowest to highest."""

    levels: list[Any]


def question(spec: Any, instructions: Any = None) -> dict:
    """A Python type (or explicit spec) -> a /v1/systemone question.

    bool -> noul (yes/no) · Literal["a", "b"], an Enum, a list/tuple of names or a dict name -> description -> choice ·
    Scale([...]) -> score · a dict with "type" is passed through."""
    if isinstance(spec, dict) and "type" in spec:
        q = dict(spec)
    elif spec is bool:
        q = {"type": "noul"}
    elif isinstance(spec, Scale):
        q = {"type": "score", "criteria": list(spec.levels)}
    elif typing.get_origin(spec) is Literal:
        q = {"type": "choice", "criteria": {str(v): None for v in typing.get_args(spec)}}
    elif isinstance(spec, type) and issubclass(spec, enum.Enum):
        q = {"type": "choice", "criteria": {str(m.value): None for m in spec}}
    elif isinstance(spec, dict):
        q = {"type": "choice", "criteria": {str(k): v for k, v in spec.items()}}
    elif isinstance(spec, (list, tuple)):
        q = {"type": "choice", "criteria": {str(v): None for v in spec}}
    else:
        raise TypeError(f"can't turn {spec!r} into a question: use bool, Literal[...], an Enum, a list, a dict or Scale")
    if instructions is not None:
        q["instructions"] = instructions
    return q


@dataclass
class Decision:
    """One answer. `value` is True/False (noul), the chosen name (choice) or the expected level (score)."""

    type: str
    value: Any
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    tier: str | None = None
    decided: bool | None = None  # with max_error: the answer is within the requested error rate
    set: list[str] | None = None  # with max_error: the options that contain the truth at 1 - max_error
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_answer(cls, a: dict) -> Decision:
        if a["type"] == "noul":
            p = a.get("noul", a.get("probability"))
            value, probs = p >= 0.5, {"false": 1 - p, "true": p}
        elif a["type"] == "choice":
            value, probs = a["choice"], a.get("probabilities", {})
        else:
            value, probs = a["score"], a.get("probabilities", {})
        return cls(a["type"], value, a.get("confidence", 0.0), probs, a.get("tier"), a.get("decided"), a.get("set"), a)

    def __bool__(self) -> bool:
        if self.type != "noul":
            raise TypeError(f"a {self.type} decision has no truth value; use .value")
        return bool(self.value)


class Decisions(dict):
    """Answers by question name, also as attributes (d.urgent), plus the raw response."""

    def __init__(self, response: dict):
        super().__init__({k: Decision.from_answer(a) for k, a in response.get("answers", {}).items()})
        self.response = response
        self.latency_ms = response.get("latency_ms")
        self.specialist = response.get("specialist")

    def __getattr__(self, name: str) -> Decision:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _body(state: Any, questions: dict[str, Any], instructions: dict[str, Any] | None, model: str | None,
          max_error: float | None) -> dict:
    instructions = instructions or {}
    body: dict[str, Any] = {"state": state,
                            "questions": {k: question(v, instructions.get(k)) for k, v in questions.items()}}
    if model:
        body["model"] = model
    if max_error is not None:
        body["max_error"] = max_error
    return body


def _check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise DragonflyError(r.status_code, detail)
    return r.json()


def _events(lines) -> Iterator[tuple[str, dict]]:
    name, data = None, []
    for line in lines:
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif not line and name:
            yield name, json.loads("".join(data))
            name, data = None, []


def _as_decision_fn(client, fn: Callable, model: str | None, max_error: float | None):
    """@decision: the docstring is the question, the return annotation its type, the first argument the state."""
    hints = typing.get_type_hints(fn, include_extras=True)
    spec = hints.get("return")
    if spec is None:
        raise TypeError(f"{fn.__name__} needs a return annotation (bool, Literal[...], Scale via Annotated, ...)")
    if typing.get_origin(spec) is typing.Annotated:
        spec = typing.get_args(spec)[1]
    q = question(spec, inspect.getdoc(fn) or fn.__name__.replace("_", " "))
    params = list(inspect.signature(fn).parameters)
    return q, params, fn.__name__


class Dragonfly:
    """Synchronous client. base_url is the server root (it serves /v1/...)."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None, timeout: float = 30.0,
                 model: str | None = None, transport: httpx.BaseTransport | None = None):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)
        self.model = model

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Dragonfly:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def decide(self, state: Any, *, instructions: dict[str, Any] | None = None, model: str | None = None,
               max_error: float | None = None, **questions: Any) -> Decisions:
        body = _body(state, questions, instructions, model or self.model, max_error)
        return Decisions(_check(self.http.post("/v1/systemone", json=body)))

    def batch(self, states: list[Any], *, instructions: dict[str, Any] | None = None, model: str | None = None,
              max_error: float | None = None, **questions: Any) -> list[Decisions | DragonflyError]:
        """The same questions over many states in one call (/v1/batch, batch priority on the server)."""
        requests = [_body(s, questions, instructions, model or self.model, max_error) for s in states]
        out = _check(self.http.post("/v1/batch", json={"requests": requests}))
        return [DragonflyError(422, r["error"]) if "error" in r else Decisions(r) for r in out["results"]]

    def stream(self, state: Any, *, instructions: dict[str, Any] | None = None, model: str | None = None,
               max_error: float | None = None, **questions: Any) -> Iterator[tuple[str, Decisions]]:
        """Anytime answers: yields ("answers", fast), maybe ("update", upgraded questions only), then ("done", final)."""
        body = _body(state, questions, instructions, model or self.model, max_error)
        with self.http.stream("POST", "/v1/systemone", params={"stream": "true"}, json=body) as r:
            if r.status_code >= 400:
                r.read()
                _check(r)
            for name, data in _events(r.iter_lines()):
                if name == "error":
                    raise DragonflyError(data.get("status", 500), data.get("detail"))
                yield name, Decisions(data)

    def decision(self, fn: Callable | None = None, *, model: str | None = None, max_error: float | None = None):
        """Decorator: turn a typed, documented function into a decision (see the module docstring)."""
        def wrap(f: Callable):
            q, params, name = _as_decision_fn(self, f, model, max_error)

            @functools.wraps(f)
            def call(*args, **kwargs) -> Decision:
                bound = dict(zip(params, args), **kwargs)
                state = bound[params[0]] if len(params) == 1 else bound
                body = {"state": state, "questions": {name: q}}
                if model or self.model:
                    body["model"] = model or self.model
                if max_error is not None:
                    body["max_error"] = max_error
                return Decisions(_check(self.http.post("/v1/systemone", json=body)))[name]
            return call
        return wrap(fn) if fn is not None else wrap


class AsyncDragonfly:
    """asyncio client with the same methods (decide, batch, stream)."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None, timeout: float = 30.0,
                 model: str | None = None, transport: httpx.AsyncBaseTransport | None = None):
        headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout,
                                      transport=transport)
        self.model = model

    async def aclose(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> AsyncDragonfly:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def decide(self, state: Any, *, instructions: dict[str, Any] | None = None, model: str | None = None,
                     max_error: float | None = None, **questions: Any) -> Decisions:
        body = _body(state, questions, instructions, model or self.model, max_error)
        return Decisions(_check(await self.http.post("/v1/systemone", json=body)))

    async def batch(self, states: list[Any], *, instructions: dict[str, Any] | None = None, model: str | None = None,
                    max_error: float | None = None, **questions: Any) -> list[Decisions | DragonflyError]:
        requests = [_body(s, questions, instructions, model or self.model, max_error) for s in states]
        out = _check(await self.http.post("/v1/batch", json={"requests": requests}))
        return [DragonflyError(422, r["error"]) if "error" in r else Decisions(r) for r in out["results"]]

    async def stream(self, state: Any, *, instructions: dict[str, Any] | None = None, model: str | None = None,
                     max_error: float | None = None, **questions: Any) -> AsyncIterator[tuple[str, Decisions]]:
        body = _body(state, questions, instructions, model or self.model, max_error)
        async with self.http.stream("POST", "/v1/systemone", params={"stream": "true"}, json=body) as r:
            if r.status_code >= 400:
                await r.aread()
                _check(r)
            lines = [line async for line in r.aiter_lines()]
            for name, data in _events(lines):
                if name == "error":
                    raise DragonflyError(data.get("status", 500), data.get("detail"))
                yield name, Decisions(data)
