"""Images and audio inside a request's `state`.

Any JSON object in the state shaped {"type": "image" | "audio", "data": "<base64>"} is sent to the perception-service
(one batched call per request) and replaced by its text form before plugins and the decision model run:

  {"type": "image", "data": ...}  ->  {"type": "image", "text": "<OCR text>", "shows": "car (0.61), damaged car (0.22)"}
  {"type": "audio", "data": ...}  ->  {"type": "audio", "transcript": "<speech as text>"}

Other keys on the object (e.g. "name", "tasks", "language") are kept or passed through. Only inline base64 is accepted;
URLs are never fetched (no server-side request forgery).
"""

from __future__ import annotations

import copy
from typing import Any

import httpx

MEDIA_TYPES = ("image", "audio")
PASS_THROUGH = ("tasks", "language", "ocr_lang")


class MediaError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def is_media(v: Any) -> bool:
    return isinstance(v, dict) and v.get("type") in MEDIA_TYPES and isinstance(v.get("data"), str)


def find_media(state: Any, path: tuple = ()) -> list[tuple[tuple, dict]]:
    """-> [(path, media object)] in document order. A path is a tuple of dict keys / list indexes."""
    if is_media(state):
        return [(path, state)]
    found = []
    if isinstance(state, dict):
        for k, v in state.items():
            found += find_media(v, path + (k,))
    elif isinstance(state, list):
        for i, v in enumerate(state):
            found += find_media(v, path + (i,))
    return found


def _replace(state: Any, path: tuple, value: Any) -> Any:
    if not path:
        return value
    parent = state
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return state


def text_form(media: dict, result: dict) -> dict:
    keep = {k: v for k, v in media.items() if k not in ("data", *PASS_THROUGH)}
    if media["type"] == "audio":
        return {**keep, "transcript": result.get("transcript", "")}
    out = dict(keep)
    if "ocr" in result:
        out["text"] = result["ocr"]["text"]
    if "tags" in result:
        out["shows"] = ", ".join(f"{t['label']} ({t['score']:.2f})" for t in result["tags"])
    return out


class MediaResolver:
    def __init__(self, url: str, timeout: float = 30.0, client: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=timeout)

    async def perceive(self, payload: dict) -> dict:
        """Raw call to the perception-service (also used by POST /v1/perceive)."""
        try:
            resp = await self.client.post(f"{self.url}/v1/perceive", json=payload)
        except httpx.HTTPError as e:
            raise MediaError(f"perception-service unreachable: {e}", 503) from e
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if resp.status_code != 200:
            raise MediaError(body.get("detail") or f"perception-service error {resp.status_code}", resp.status_code)
        return body

    async def resolve(self, state: Any) -> tuple[Any, float]:
        """-> (state with every media object replaced by text, perception milliseconds). No media: unchanged, 0."""
        found = find_media(state)
        if not found:
            return state, 0.0
        items = []
        for n, (_, media) in enumerate(found):
            item = {"id": str(n), "type": media["type"], "data": media["data"]}
            item.update({k: media[k] for k in PASS_THROUGH if k in media})
            items.append(item)
        body = await self.perceive({"items": items})
        state = copy.deepcopy(state)
        for n, (path, media) in enumerate(found):
            state = _replace(state, path, text_form(media, body["items"][str(n)]))
        return state, float(body.get("latency_ms", {}).get("total", 0.0))

    async def close(self) -> None:
        await self.client.aclose()
