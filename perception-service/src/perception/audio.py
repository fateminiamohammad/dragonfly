"""Speech -> text with CTC models: non-autoregressive. The model labels every audio frame in parallel in one forward
pass; the transcript is the frame labels with repeats collapsed and blanks removed (greedy CTC). No token-by-token
generation, unlike Whisper.
"""

from __future__ import annotations

import io
import logging
import subprocess
import wave

import numpy as np
import torch

from .device import gpu_dtype

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000


def decode_audio(data: bytes, max_seconds: float) -> np.ndarray:
    """Any audio file -> float32 mono at 16 kHz. WAV/FLAC/OGG/MP3 are decoded in-process (libsndfile); anything else
    (m4a, opus, webm...) through ffmpeg. Raises ValueError for undecodable or too-long audio."""
    wav = _decode_soundfile(data)
    if wav is None and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        wav = _decode_wav(data)
    if wav is None:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                 "-ar", str(SAMPLE_RATE), "pipe:1"],
                input=data, capture_output=True, timeout=60, check=False)
        except FileNotFoundError as e:
            raise ValueError("only WAV is supported without ffmpeg installed") from e
        if proc.returncode != 0 or not proc.stdout:
            raise ValueError(f"could not decode audio: {proc.stderr.decode(errors='replace')[:200]}")
        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        raise ValueError("audio is empty")
    seconds = wav.size / SAMPLE_RATE
    if seconds > max_seconds:
        raise ValueError(f"audio is {seconds:.1f} s; the limit is {max_seconds:.0f} s")
    return wav


def _resample(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == SAMPLE_RATE:
        return x.astype(np.float32)
    n = int(round(x.size * SAMPLE_RATE / rate))
    return np.interp(np.linspace(0, x.size - 1, n), np.arange(x.size), x).astype(np.float32)


def _decode_soundfile(data: bytes) -> np.ndarray | None:
    try:
        import soundfile
    except ImportError:
        return None
    try:
        x, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        return None  # not a format libsndfile knows: let the next decoder try
    return _resample(x.mean(axis=1), rate)


def _decode_wav(data: bytes) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    except wave.Error as e:
        raise ValueError(f"invalid WAV: {e}") from e
    if width == 1:
        x = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported WAV sample width {width * 8} bits")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return _resample(x, rate)


def ctc_collapse(ids: list[int], blank: int) -> list[int]:
    """Greedy CTC: merge repeated frame labels, then drop blanks. [a a _ a b b _] -> [a a b]."""
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


class CTCRecognizer:
    """One CTC speech model (e.g. Parakeet-CTC for English, wav2vec2-XLSR for Persian)."""

    def __init__(self, model_id: str, device: str, model=None, processor=None):
        from transformers import AutoModelForCTC, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        dtype = gpu_dtype(device)
        self.model = (model or AutoModelForCTC.from_pretrained(model_id, dtype=dtype)).to(device).eval()
        self.blank = self.model.config.pad_token_id if self.model.config.pad_token_id is not None else 0
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    @torch.inference_mode()
    def transcribe(self, waves: list[np.ndarray]) -> list[str]:
        inputs = self.processor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        dtype = next(self.model.parameters()).dtype
        inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        frames = logits.argmax(-1).cpu().tolist()
        texts = []
        for ids in frames:
            tokens = ctc_collapse(ids, self.blank)
            try:  # wav2vec2 tokenizers would group repeats again; we already did
                text = self.tokenizer.decode(tokens, group_tokens=False, skip_special_tokens=True)
            except TypeError:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            texts.append(" ".join(text.split()))
        return texts
