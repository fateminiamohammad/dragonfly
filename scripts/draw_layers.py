"""Draw the layer stacks of every model Dragonfly runs (docs/images/layers-*.svg).

  python scripts/draw_layers.py

Numbers come from each model's config.json on the Hugging Face Hub (checked 2026-09-29); re-check them when a
backbone changes. Plain SVG, no dependencies, same style as docs/images/tier-m-mask.svg.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
INK, MUTED = "#16181d", "#5d6470"
BLUE = ("#dbe7fb", "#3b6fd8")      # inputs / embeddings
TEAL = ("#d8f0ec", "#0f7b6c")      # transformer layers
TEAL2 = ("#b9e3db", "#0f7b6c")     # global-attention layers
GREY = ("#eceef2", "#8a919c")      # frozen
ORANGE = ("#fdebd3", "#c77700")    # trained (LoRA, heads)
PURPLE = ("#ece3fb", "#6b46c1")    # output


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                      'font-family="Segoe UI, Helvetica, Arial, sans-serif">',
                      f"  <title>{escape(title)}</title>", f'  <rect width="{w}" height="{h}" fill="#ffffff"/>']

    def text(self, x, y, s, size=12.5, color=INK, anchor="start", weight="normal"):
        self.parts.append(f'  <text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
                          f'font-weight="{weight}">{escape(s)}</text>')

    def box(self, x, y, w, h, colors, label="", sub="", size=12.5):
        fill, stroke = colors
        self.parts.append(f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>')
        if label:
            cy = y + h / 2 + (-3 if sub else 4.5)
            self.text(x + w / 2, cy, label, size, anchor="middle")
        if sub:
            self.text(x + w / 2, y + h / 2 + 13, sub, 11, MUTED, "middle")

    def arrow(self, x, y1, y2):
        self.parts.append(f'  <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 5}" stroke="{MUTED}" stroke-width="1.4"/>')
        self.parts.append(f'  <path d="M{x - 4},{y2 - 6} L{x},{y2} L{x + 4},{y2 - 6} Z" fill="{MUTED}"/>')

    def stack(self, x, y, w, n, height, pick, label):
        """n thin bars (one per layer), colored by pick(i), with a bracket label."""
        bar = height / n
        for i in range(n):
            fill, stroke = pick(i)
            self.parts.append(f'  <rect x="{x}" y="{y + i * bar:.1f}" width="{w}" height="{bar - 1.2:.1f}" rx="1.5" '
                              f'fill="{fill}" stroke="{stroke}" stroke-width="0.6"/>')
        self.parts.append(f'  <path d="M{x + w + 6},{y} h5 v{height} h-5" fill="none" stroke="{MUTED}"/>')
        self.text(x + w + 16, y + height / 2 + 4, label, 12, MUTED)

    def save(self, name):
        self.parts.append("</svg>\n")
        (OUT / name).write_text("\n".join(self.parts), encoding="utf-8")
        print("wrote", OUT / name)


def column(s: Svg, x, w, blocks, y=70):
    """blocks: list of ("box", colors, label, sub, h) or ("stack", n, pick, label, h, detail_lines)."""
    for i, b in enumerate(blocks):
        if b[0] == "box":
            _, colors, label, sub, h = b
            s.box(x, y, w, h, colors, label, sub)
        else:
            _, n, pick, label, h, detail = b
            s.stack(x, y, w * 0.45, n, h, pick, label)
            for j, line in enumerate(detail):
                s.text(x + w * 0.45 + 16, y + h / 2 + 22 + j * 15, line, 11, MUTED)
        y += b[-1] if b[0] == "box" else b[4]
        if i < len(blocks) - 1:
            s.arrow(x + w / 2 if b[0] == "box" else x + w * 0.225, y + 2, y + 20)
            y += 22
    return y


def decision():
    s = Svg(980, 720, "Dragonfly decision models: layer by layer")
    s.text(24, 34, "Decision models: every layer, one forward pass", 17, weight="600")
    s.text(24, 54, "numbers from each model's config.json · teal = transformer layers · orange = trained by Dragonfly "
                   "· grey = frozen", 12, MUTED)

    # tier S: ModernBERT-base (fully fine-tuned)
    x, w = 24, 440
    s.text(x, 92, "Tier S · ModernBERT-base (149M, all weights trained)", 14, weight="600")
    column(s, x, w, [
        ("box", BLUE, "[CLS] question · options · [SEP] state", "one row per question · up to 1,536 tokens", 44),
        ("box", BLUE, "token embeddings 50,368 × 768", "RoPE positions, no learned position table", 40),
        ("stack", 22, lambda i: TEAL2 if i % 3 == 0 else TEAL, "22 layers", 250,
         ["pre-norm LayerNorm → attention", "12 heads × 64", "dark: global attention (every 3rd)",
          "light: local, 128-token window", "GeGLU MLP 768 → 2×1,152 → 768", "unpadded (no work on padding)"]),
        ("box", ORANGE, "pointer head: q = W·h[question], k = W·h[option]", "768 → 256 · LayerNorm · q·k / √256", 44),
        ("box", PURPLE, "softmax over the options / T", "calibrated probabilities", 40),
    ], y=106)

    # tier M: Qwen3-4B-Base + LoRA
    x, w = 510, 446
    s.text(x, 92, "Tier M · Qwen3-4B-Base (frozen) + LoRA r=16", 14, weight="600")
    column(s, x, w, [
        ("box", BLUE, "[state][q1][opts…][q2][opts…] packed", "custom 4D mask · options share position ids", 44),
        ("box", GREY, "token embeddings 151,936 × 2,560", "RoPE positions (θ from config)", 40),
        ("stack", 36, lambda i: GREY, "36 layers", 250,
         ["RMSNorm → grouped-query attention", "32 query / 8 key-value heads × 128", "QK-RMSNorm · RoPE",
          "RMSNorm → SwiGLU MLP 2,560 → 9,728 → 2,560", "LoRA (orange, r=16) beside q k v o",
          "and gate up down: 7 per layer, 252 total"]),
        ("box", ORANGE, "pointer head: last token of question vs option", "2,560 → 256 · LayerNorm · q·k / √256", 44),
        ("box", PURPLE, "softmax over the options / T", "order-invariant by construction", 40),
    ], y=106)
    # LoRA marks on the M stack
    for i in range(36):
        y = 106 + 44 + 22 + 40 + 22 + i * (250 / 36)
        s.parts.append(f'  <rect x="{510 + 446 * 0.45 - 12}" y="{y + 1:.1f}" width="8" height="{250 / 36 - 3.2:.1f}" '
                       f'rx="1" fill="{ORANGE[0]}" stroke="{ORANGE[1]}" stroke-width="0.6"/>')
    s.text(24, 700, "Tier S answers first; questions below the confidence threshold (0.70) are asked again to tier M. "
                    "Both heads score all options in parallel: no tokens are generated.", 12, MUTED)
    s.save("layers-decision.svg")


def perception():
    s = Svg(1000, 580, "Dragonfly perception models: layer by layer")
    s.text(24, 34, "Perception models: images and audio → text, also non-autoregressive", 17, weight="600")
    s.text(24, 54, "each predicts all its outputs at once (CTC or one embedding) · numbers from each model's config",
           12, MUTED)
    cols = [
        ("Speech (English) · Parakeet-CTC 0.6B", [
            ("box", BLUE, "16 kHz audio → log-mel", "10 ms per frame", 40),
            ("box", BLUE, "conv subsampling ×8", "one vector per 80 ms", 40),
            ("stack", 24, lambda i: TEAL, "24 layers", 200, ["FastConformer", "1,024 · 8 heads", "FFN 4,096", "+ conv module"]),
            ("box", PURPLE, "CTC: a symbol per step", "collapse repeats, drop blanks", 40),
        ]),
        ("Speech (Persian) · wav2vec2 XLSR-53", [
            ("box", BLUE, "16 kHz raw waveform", "no spectrogram", 40),
            ("box", BLUE, "7 conv layers, 512 ch", "one vector per 20 ms", 40),
            ("stack", 24, lambda i: TEAL, "24 layers", 200, ["transformer", "1,024 · 16 heads", "FFN 4,096"]),
            ("box", PURPLE, "CTC over 67 characters", "collapse repeats, drop blanks", 40),
        ]),
        ("Image tags · SigLIP-2 ViT-B/16", [
            ("box", BLUE, "224 × 224 image", "16 × 16 pixel patches", 40),
            ("box", BLUE, "196 patch tokens × 768", "+ position embeddings", 40),
            ("stack", 12, lambda i: TEAL, "12 layers", 200, ["ViT encoder", "768 · 12 heads", "MLP 3,072"]),
            ("box", PURPLE, "image · tag-text embeddings", "every tag scored at once", 40),
        ]),
        ("Text in images · PP-OCRv4", [
            ("box", BLUE, "image (long side ≤ 960 px)", "", 40),
            ("box", BLUE, "DBNet text detection", "probability map → line boxes", 40),
            ("box", TEAL, "SVTR recognizer", "per text line, lines batched", 200),
            ("box", PURPLE, "CTC per line", "right-to-left order for Persian", 40),
        ]),
    ]
    for k, (title, blocks) in enumerate(cols):
        x = 20 + k * 245
        s.text(x, 92, title, 12.5, weight="600")
        column(s, x, 225, blocks, y=106)
    s.text(24, 560, "The text they produce replaces the image or audio in the request's state; the decision model then "
                    "answers in its usual single pass.", 12, MUTED)
    s.save("layers-perception.svg")


if __name__ == "__main__":
    decision()
    perception()
