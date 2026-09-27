import base64
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from perception.api import Perception, create_app
from perception.audio import SAMPLE_RATE, ctc_collapse, decode_audio
from perception.ocr import Line, is_rtl, join_text, reading_order


# ---- pure logic --------------------------------------------------------------------------------------------------
def test_ctc_collapse_merges_repeats_and_drops_blanks():
    # frames: h h _ e l l _ l o  (blank = 0) -> h e l l o
    assert ctc_collapse([5, 5, 0, 2, 7, 7, 0, 7, 9], blank=0) == [5, 2, 7, 7, 9]
    assert ctc_collapse([0, 0, 0], blank=0) == []


def wav_bytes(seconds: float, rate: int = 8000, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    x = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype("<i2")
    if channels == 2:
        x = np.repeat(x[:, None], 2, axis=1).reshape(-1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def test_wav_is_resampled_to_16k_mono():
    wav = decode_audio(wav_bytes(1.0, rate=8000, channels=2), max_seconds=60)
    assert wav.dtype == np.float32 and abs(wav.size - SAMPLE_RATE) <= 1
    assert 0.4 < np.abs(wav).max() <= 0.51


def test_audio_limits():
    with pytest.raises(ValueError, match="limit"):
        decode_audio(wav_bytes(3.0), max_seconds=2)
    with pytest.raises(ValueError):
        decode_audio(b"RIFF\x00\x00\x00\x00WAVEgarbage", max_seconds=60)


def line(text, x, y, w=100, h=20):
    return Line(text, 0.9, [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


def test_reading_order_ltr_rows():
    rows = reading_order([line("world", 200, 10), line("second row", 0, 60), line("hello", 0, 12)])
    assert join_text(rows) == "hello world\nsecond row"


def test_reading_order_rtl_row_reads_right_to_left():
    # Persian: the right-most box is read first
    rows = reading_order([line("سلام", 300, 10), line("دنیا", 100, 10)])
    assert join_text(rows) == "سلام دنیا"
    assert is_rtl("سلام دنیا") and not is_rtl("hello")


# ---- API with stub models (no downloads) -------------------------------------------------------------------------
class StubTagger:
    def tag(self, images, top_k=8, min_p=0.05):
        return [[{"label": "car", "score": 0.61, "p": 0.93}] for _ in images]


class StubOCR:
    def read(self, image):
        return {"text": f"{image.width}x{image.height}", "lines": []}


class StubASR:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, waves):
        self.calls += 1
        return [self.text for _ in waves]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def png(w=32, h=16) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    en, fa = StubASR("hello"), StubASR("سلام")
    p = Perception("cpu", {"en": "x", "fa": "y"}, "t", ["car"], ["en"], max_mb=1, max_audio_s=10,
                   tagger=StubTagger(), ocr={"en": StubOCR()}, recognizers={"en": en, "fa": fa})
    with TestClient(create_app(p)) as c:
        c.asr = {"en": en, "fa": fa}
        yield c


def test_image_and_audio_in_one_call(client):
    r = client.post("/v1/perceive", json={"items": [
        {"id": "photo", "type": "image", "data": b64(png())},
        {"id": "a1", "type": "audio", "data": b64(wav_bytes(1.0)), "language": "en"},
        {"id": "a2", "type": "audio", "data": b64(wav_bytes(1.0)), "language": "en"},
        {"id": "a3", "type": "audio", "data": b64(wav_bytes(1.0)), "language": "fa"},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"]["photo"] == {"type": "image", "tags": [{"label": "car", "score": 0.61, "p": 0.93}],
                                      "ocr": {"text": "32x16", "lines": []}}
    assert body["items"]["a1"]["transcript"] == "hello" and body["items"]["a3"]["transcript"] == "سلام"
    assert body["items"]["a1"]["duration_s"] == 1.0
    assert client.asr["en"].calls == 1  # both English clips in one batch
    assert {"decode", "tags", "ocr", "asr", "total"} <= set(body["latency_ms"])


def test_tasks_select_work(client):
    r = client.post("/v1/perceive", json={"items": [{"id": "p", "type": "image", "data": b64(png()), "tasks": ["tags"]}]})
    assert "ocr" not in r.json()["items"]["p"]


@pytest.mark.parametrize("item,needle", [
    ({"id": "x", "type": "image", "data": "not base64!!"}, "base64"),
    ({"id": "x", "type": "image", "data": b64(b"not an image")}, "decode image"),
    ({"id": "x", "type": "audio", "data": b64(wav_bytes(1.0)), "language": "de"}, "no speech model"),
    ({"id": "x", "type": "audio", "data": b64(wav_bytes(11.0))}, "limit"),
    ({"id": "x", "type": "image", "data": b64(png()), "ocr_lang": "arabic"}, "not enabled"),
    ({"id": "x", "type": "image", "data": "A" * 2_000_000}, "larger than"),
])
def test_bad_items_are_422(client, item, needle):
    r = client.post("/v1/perceive", json={"items": [item]})
    assert r.status_code == 422 and needle in r.text


def test_duplicate_ids_rejected(client):
    item = {"id": "same", "type": "image", "data": b64(png())}
    assert client.post("/v1/perceive", json={"items": [item, item]}).status_code == 422
