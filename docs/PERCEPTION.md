# Perception: images and audio

Dragonfly decides on text. The **perception-service** turns images and audio into text first, using only
**non-autoregressive** models: one forward pass each, with no token-by-token generation. The decision stays as fast as
the rest of Dragonfly.

| Input | Output | Model | Why it's one pass | License |
|---|---|---|---|---|
| Speech (English) | transcript | `nvidia/parakeet-ctc-0.6b` | CTC: every audio frame is labelled in parallel, then repeats and blanks are collapsed | CC-BY-4.0 |
| Speech (Persian) | transcript | `jonatasgrosman/wav2vec2-large-xlsr-53-persian` | CTC | Apache-2.0 |
| Text in images | OCR text + lines + boxes | PP-OCRv4 via RapidOCR (torch engine): DBNet detection + CTC recognition | segmentation + CTC | Apache-2.0 |
| What an image shows | ranked tags | `google/siglip2-base-patch16-224` against a tag vocabulary | one image embedding and one matrix multiply | Apache-2.0 |

**Not used on purpose:** Whisper, Florence-2/BLIP captions and GPT-4o vision are autoregressive. A generated caption
could later be added as an opt-in plugin, clearly labelled as slow.

## Using it

### Inside a decision request

Put media objects anywhere in `state`:

```json
{
  "state": {
    "message": "see attached",
    "photo": { "type": "image", "data": "<base64>", "name": "bumper.jpg" },
    "voice": { "type": "audio", "data": "<base64>", "language": "fa" }
  },
  "questions": { "damaged": { "type": "noul", "instructions": "Is the car damaged?" } }
}
```

**What happens:**
1. The model-service sends all media in one batched call.
2. It replaces each object with text: an image becomes `{"type": "image", "name": ..., "text": "<OCR>", "shows":
   "car (0.41), damaged car (0.22), ..."}`, and audio becomes `{"type": "audio", "transcript": "..."}`.
3. Then plugins and the decision model run on that text.

The response reports perception time separately in `usage.media_ms`. Repeated identical media hit the answer cache.

**Optional keys on a media object:**

| Key | Applies to | Values |
|---|---|---|
| `tasks` | images | `["ocr", "tags"]` (the default), or just one of them |
| `ocr_lang` | images | `en` or `arabic` (Persian, Arabic, Urdu) |
| `language` | audio | `en` or `fa` |

Other keys, such as `name`, are kept.

### Just convert: `POST /v1/perceive`

The same API-key auth as `/v1/systemone`. Up to 16 items per call:

```json
{"items": [
  {"id": "scan",  "type": "image", "data": "<base64>", "tasks": ["ocr"], "ocr_lang": "en"},
  {"id": "voice", "type": "audio", "data": "<base64>", "language": "en"}
]}
```

It returns `items[id]` with `ocr` (`text`, `lines[{text, confidence, box}]`), `tags` (`label`, `score`, `p`) or
`transcript` (`language`, `duration_s`), plus `latency_ms` per stage.

The UI's **Media** page does this with a file picker. The Playground's **Attach image/audio** button inserts a media
object into the request.

### Limits

**Formats:**
- Images: anything Pillow reads.
- Audio: WAV, FLAC, OGG and MP3 in-process; m4a, opus and webm through ffmpeg (in the image).

**Size and safety:**
- 20 MB per item (`PERCEPTION_MAX_MB`) and 60 s of audio (`PERCEPTION_MAX_AUDIO_S`).
- Base64 only. **URLs are never fetched**, which prevents server-side request forgery (SSRF).
- Bad media returns 422 with the item's id.

## Tags: what the numbers mean

- **`score`:** the tag's **share** among all tags (a softmax over the vocabulary). Use it to rank; it is relative,
  not a probability.
- **`p`:** SigLIP's own sigmoid output. It is **not calibrated** for a large generic vocabulary. A correct "cat"
  scored 0.003 while an invoice scored 0.87, so don't threshold on `p`.

**Vocabulary:**
- The default is 216 tags aimed at decisions: documents, vehicles and damage, packages, safety, image quality
  and more.
- Swap it for your own domain with `PERCEPTION_TAGS_FILE` (one tag per line). Tag text embeddings are computed once at
  startup, so a longer vocabulary costs almost nothing per image.

## Measured (RTX 3090 Ti, bf16, warm)

| Stage | Input | Result | Time |
|---|---|---|---|
| Speech → text | 10.4 s LibriSpeech clip | exact reference transcript (28/28 words) | **79 ms** (about 130× real time) |
| Speech → text, 1.1B model | same | same text | 131 ms |
| OCR | rendered 900×260 invoice | exact: `INVOICE #4821 / Total due: $1,250.00 / Due date: 2026-10-15` | **66 ms** |
| OCR | 640×480 photo, no text | no text | 41 ms |
| Tags (CUDA graph) | COCO "two cats on a sofa" photo | cat 0.33, sofa 0.10, television 0.07, … | **11 ms** |
| Tags | rendered invoice | invoice 0.98 | 11 ms |

**Versus common autoregressive models** (same GPU; `bench/compare_media.py`; p50 of 5 warm runs):

| Task | Dragonfly (perception + decision) | Common model | Dragonfly is | Correct |
|---|---|---|---|---|
| Invoice image → document type | **103 ms** | Qwen2.5-VL-7B: 2,233 ms (answer only), 3,049 ms (reasoning) | **22–30× faster** | both 100% |
| Cats photo → picture type | **71 ms** | Qwen2.5-VL-7B: 2,306 / 2,771 ms | **32–39× faster** | both 100% |
| Invoice image → total amount due | **104 ms** | Qwen2.5-VL-7B: 2,336 / 2,414 ms | **22–23× faster** | both 100% |
| 10.4 s speech → text | **71 ms** | Whisper large-v3-turbo (in-process): 551 ms | **7.8× faster** | same words |

End to end through nginx (`scripts/e2e_media.py`), one request with an invoice image and a voice message correctly
answered:
- the document type (invoice);
- the amount, read by OCR ($1,250.00);
- whether the voice message talks about food (yes).

**Engineering fixes behind these numbers:**
- **bf16, not fp16:** `parakeet-ctc-0.6b` overflowed to NaN in fp16.
- **OCR detection capped at 960 px on the longer side:** RapidOCR's default raised the shorter side to 736 px, which
  made detection 2× slower for the same text.
- **CUDA graphs for the SigLIP image model:** 24 → 11 ms.
- **Warm-up at startup:** the first real request took 3.7 s before this.

**Persian speech:** the model reports 30.1% word error rate and 7.4% character error rate on Common Voice Persian.
That's good enough to drive decisions, not to produce verbatim transcripts. We have not yet measured it on our own
Persian audio.

## Configuration

See the `perception-service` section in `docker/.env.example`:
- `PERCEPTION_URL` (on the model-service);
- `PERCEPTION_ASR_MODEL_EN`, `PERCEPTION_ASR_MODEL_FA` and `PERCEPTION_ASR_MODELS` (more languages, `lang=model`);
- `PERCEPTION_TAGS_MODEL`, `PERCEPTION_TAGS_FILE`, `PERCEPTION_OCR_LANGS`, `PERCEPTION_PRELOAD`;
- `PERCEPTION_MAX_MB`, `PERCEPTION_MAX_AUDIO_S`.

The first start downloads about 3 GB of models into the shared `hfcache` volume. Later starts load from disk.
