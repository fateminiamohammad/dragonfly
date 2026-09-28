# How Dragonfly works

This page explains the algorithm with pictures. The code lives in `model-service/src/dragonfly/` and
`perception-service/src/perception/`.

## 1. The core idea: answer every question in one pass

An LLM answers by **writing**: it produces one token, feeds it back in, produces the next, and repeats. Often it writes
reasoning first. Dragonfly never writes. The question's possible answers are already known (the `criteria`), so it
**scores all of them at once** in a single forward pass and returns a probability for each.

```mermaid
flowchart LR
    subgraph LLM["LLM (autoregressive)"]
        direction LR
        A1[read prompt] --> A2[token 1] --> A3[token 2] --> A4[token 3] --> A5["… token N<br/>(reasoning + JSON)"] --> A6[parse answer]
    end
    subgraph DF["Dragonfly (one pass)"]
        direction LR
        B1["read state + questions + all options<br/>together"] --> B2["score every option<br/>at once"] --> B3["probabilities<br/>for every answer"]
    end
```

**Measured cost of that difference** (same GPU): a thinking LLM took 1,658 ms per request and Dragonfly took 16 ms.

## 2. What happens to one request

```mermaid
flowchart TD
    C([client: POST /v1/systemone]) --> N[nginx]
    N --> K{API key valid?<br/>in-process cache → Redis}
    K -- no --> E401([401])
    K -- yes --> M{images or audio<br/>in the state?}
    M -- yes --> P["perception-service<br/>speech → text · OCR · image tags<br/>(one batched call)"]
    P --> T[state is now all text]
    M -- no --> T
    T --> PL["plugins: on_request<br/>(validate, redact, enrich)"]
    PL --> CA{same request<br/>answered before?}
    CA -- yes --> R
    CA -- no --> S["tier S: fast model<br/>answers every question"]
    S --> Q{confidence ≥ 0.45?}
    Q -- yes --> R["answers + probabilities"]
    Q -- no --> MM["tier M: quality model<br/>re-answers only the unsure questions"]
    MM --> R
    R --> PD["plugins: on_decision / on_low_confidence"]
    PD --> OUT([response + usage counted in Redis])
```

**Sizes and costs:**
- The backend (accounts, keys, usage) is **never in this path**, so it adds no latency.
- Typical cost on an RTX 3090 Ti: tier S about 8 ms, tier M about 20 ms, perception 10–80 ms per image or clip.

## 3. One primitive for every question type

Every question becomes "pick among options", so one model head serves all three types:

```mermaid
flowchart LR
    CH["choice<br/>criteria: {billing, technical, sales}"] --> O1["options:<br/>billing · technical · sales"]
    NO["noul (yes/no)"] --> O2["options:<br/>no · yes"]
    SC["score<br/>criteria: [low, medium, high]"] --> O3["options:<br/>low · medium · high"]
    O1 & O2 & O3 --> PH["pointer head<br/>probability per option"]
    PH --> A1["choice → most likely name"]
    PH --> A2["noul → p(yes)"]
    PH --> A3["score → expected level"]
```

## 4. Tier S: the fast model (ModernBERT encoder)

Each question becomes one row that the encoder reads in both directions. The options come **before** the document, so
cutting a long document can never cut an option.

```mermaid
flowchart LR
    subgraph ROW["one row per question"]
        direction LR
        R1["[CLS]"] --- R2["instructions"] --- R3["– option 1"] --- R4["– option 2"] --- R5["…"] --- R6["[SEP] state [SEP]"]
    end
    ROW --> ENC["ModernBERT-base<br/>(150M, bidirectional)"]
    ENC --> QV["query = [CLS] vector"]
    ENC --> KV["key per option =<br/>mean of its tokens"]
    QV & KV --> SCORE["score = q·k / √d<br/>÷ temperature"]
    SCORE --> SM["softmax over options<br/>→ calibrated probabilities"]
```

All questions of a request, and all requests in a batch, run as rows of **one** forward pass.

## 5. Tier M: the quality model (Qwen3 + LoRA)

A whole request is **one packed sequence**. A custom attention mask controls who can see whom:

![Tier M packed sequence and attention mask](images/tier-m-mask.svg)

**What the mask guarantees:**
- **The document is read once.** Every question and option sees it; nothing is repeated per question.
- **Questions are isolated.** Question 2 can't see question 1 or its options, so answers can't leak between them.
- **Options are isolated and share positions.** Every option of a question starts at the same position ID and can't
  see its siblings. The model therefore sees each option as if it were the only one, so **reordering options cannot
  change the scores**. The fp32 test is exact; in bf16, 0.86% of answers flip on near-ties.

The pointer head compares the question's last token (query) with each option's last token (key).

## 6. The cascade: fast first, careful only when needed

```mermaid
flowchart LR
    Q[questions] --> S["tier S<br/>≈ 8 ms"]
    S --> D{calibrated<br/>confidence}
    D -- "≥ 0.45 (64.5% of questions)" --> A([answer])
    D -- "< 0.45 (35.5%)" --> M["tier M<br/>≈ 20 ms, reads the document once"]
    M --> A
    M -. "still unsure?" .-> P["on_low_confidence plugin<br/>(e.g. escalate to an LLM or a human)"]
```

**The threshold (0.45)** is chosen on held-out calibration data (`scripts/tune_cascade.py`), not on the test set.

**Result on 1,440 test questions:**
- tier S alone: 68.2%;
- tier M alone: 76.9%;
- **cascade: 76.6%**, with 64.5% of questions never touching tier M.

## 7. Calibration: "70% sure" should mean right 70% of the time

```mermaid
flowchart LR
    TR["train with cross-entropy<br/>(a proper scoring rule)"] --> T["fit one temperature T<br/>on held-out data"]
    T --> P["serve softmax(logits / T)"]
    P --> CH["check: ECE, Brier,<br/>auto-rate at 5% error"]
```

Calibration is what makes the cascade and `on_low_confidence` work: confidence decides when to ask the bigger model.
Measured calibration error: tier S 0.032, tier M 0.027.

## 8. Distillation: the big model teaches the small one

```mermaid
flowchart LR
    TR[train split] --> M["tier M answers every training question<br/>(calibrated probabilities)"]
    M --> MIX["soft target =<br/>½ · true label + ½ · M's probabilities"]
    MIX --> S["train tier S on the soft targets"]
    S --> R["tier S: better accuracy (67.7 → 68.2%),<br/>better calibration (ECE 0.038 → 0.032)"]
```

## 9. Images and audio: turned into text, still in one pass

Only **non-autoregressive** models are used, meaning none of them generate text token by token:

```mermaid
flowchart TD
    subgraph AUDIO["audio"]
        AU[wav / mp3 / ogg / m4a] --> DEC[decode → 16 kHz mono]
        DEC --> CTC["CTC model labels every audio frame at once<br/>(Parakeet en · wav2vec2 fa)"]
        CTC --> COL["collapse repeats, drop blanks"] --> TX1["transcript"]
    end
    subgraph IMAGE["image"]
        IM[png / jpg / webp] --> DET["DBNet: find text lines"]
        DET --> REC["CTC recognizer reads every line<br/>(RTL-aware order for Persian)"] --> TX2["text in the image"]
        IM --> SIG["SigLIP-2 image embedding"]
        SIG --> MUL["× precomputed embeddings of 216 tags<br/>(one matrix multiply)"] --> TX3["what it shows: cat 0.33, sofa 0.10 …"]
    end
    TX1 & TX2 & TX3 --> STATE["replaces the media in the state"] --> DECIDE["decision models (sections 4–6)"]
```

**Measured on an RTX 3090 Ti:**
- 10.4 s of speech becomes the exact transcript in 79 ms; Whisper, which generates text, needed 551 ms.
- A decision about an invoice image takes 103 ms; a 7B vision LLM needed 2.2 s.

## 10. Why the GPU part is fast: CUDA graphs

A forward pass is thousands of tiny GPU operations, and launching each one from Python costs more than running it.

```mermaid
flowchart LR
    subgraph EAGER["eager: launch each kernel"]
        direction LR
        E1[k1] --> E2[k2] --> E3[k3] --> E4["… 3,100 launches<br/>(tier M ≈ 113 ms)"]
    end
    subgraph GRAPH["CUDA graph: record once, replay"]
        direction LR
        G1["record the whole pass<br/>per shape bucket (at startup)"] --> G2["replay = 1 launch<br/>(tier M ≈ 20 ms)"]
    end
```

**How it's done:**
- Inputs are padded to shape buckets, and padding is masked out, so answers are unchanged (tested).
- Common buckets are recorded at startup, so no request pays the recording cost.
- Tier M's LoRA adapters are merged into its weights when serving, which removes about 200 extra small multiplies per
  pass.
