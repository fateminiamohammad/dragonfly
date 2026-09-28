"""Train a Dragonfly checkpoint on labelled JSONL, fit its temperature, evaluate, save.

  dragonfly-train --train data/train.jsonl --calibration data/calibration.jsonl --test data/test.jsonl --out runs/s
  dragonfly-train --tier M --backbone Qwen/Qwen3-1.7B-Base --grad-checkpointing --batch-size 8 ... --out runs/m

Tier S fine-tunes the whole encoder. Tier M freezes the decoder and trains LoRA adapters plus the head.
Loss: cross-entropy over options (against the soft target when a question has one). Choice options are shuffled per
example so tier S cannot learn to prefer positions (tier M cannot see order at all).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from ..calibration import fit_temperature
from ..data import load_records
from ..engine import CheckpointConfig, Engine, build_model, default_device, load_backbone
from .evaluate import collect_logits, evaluate, question_rows

log = logging.getLogger("dragonfly.train")


def shuffle_options(q: dict, rng: random.Random) -> dict:
    if q["qtype"] != "choice" or len(q["options"]) < 2:
        return q
    perm = list(range(len(q["options"])))
    rng.shuffle(perm)
    out = {**q, "options": [q["options"][i] for i in perm], "label": perm.index(q["label"])}
    if "target" in q:
        out["target"] = [q["target"][i] for i in perm]
    return out


def fitting(engine: Engine, rows: list[dict], states: list[str]) -> tuple[list[dict], list[str]]:
    """Drop questions that cannot be encoded (e.g. options longer than --max-length) instead of failing mid-run."""
    keep_rows, keep_states, dropped = [], [], 0
    for q, s in zip(rows, states):
        try:
            engine.encode([{"state": s, "questions": [q]}])
        except ValueError:
            dropped += 1
            continue
        keep_rows.append(q)
        keep_states.append(s)
    if dropped:
        log.warning("skipped %d of %d questions that do not fit in --max-length %d", dropped, len(rows),
                    engine.config.max_length)
    return keep_rows, keep_states


def loss_fn(logits: torch.Tensor, rows: list[dict]) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), -1)
    losses = []
    for r, q in enumerate(rows):
        k = len(q["options"])
        if "target" in q:
            t = torch.tensor(q["target"], device=logp.device)
            losses.append(-(t * logp[r, :k]).sum())
        else:
            losses.append(-logp[r, q["label"]])
    return torch.stack(losses).mean()


def train(args) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or default_device()
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    config = CheckpointConfig(tier=args.tier, backbone=args.backbone, head_dim=args.head_dim, max_length=args.max_length,
                              lora_r=args.lora_r, lora_alpha=2 * args.lora_r, head_norm=not args.no_head_norm)
    model = build_model(config, load_backbone(args.backbone, args.tier, device))
    if args.grad_checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    engine = Engine(model, tokenizer, config, device)

    rows, states = fitting(engine, *question_rows(load_records(args.train)))
    log.info("tier %s: training on %d questions, device=%s", args.tier, len(rows), device)
    steps_per_epoch = math.ceil(len(rows) / args.batch_size)
    total = steps_per_epoch * args.epochs
    head_params = list(model.head.parameters())
    body_params = [p for p in model.backbone.parameters() if p.requires_grad]  # all of S; only LoRA for M
    log.info("trainable parameters: %.1fM", sum(p.numel() for p in head_params + body_params) / 1e6)
    params = [
        {"params": body_params, "lr": args.lr or (2e-4 if args.tier == "M" else 3e-5)},
        {"params": head_params, "lr": args.head_lr or (3e-4 if args.tier == "M" else 1e-3)},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * total), total)
    rng = random.Random(args.seed)

    step, started, ema = 0, time.time(), None
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(rows)))
        rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            batch_rows = [shuffle_options(rows[i], rng) for i in idx]
            batch, _ = engine.encode([{"state": states[i], "questions": [q]} for i, q in zip(idx, batch_rows)])
            loss = loss_fn(engine.logits(batch), batch_rows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params + body_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % args.log_every == 0:
                log.info("epoch %d step %d/%d loss %.4f (running avg %.4f) (%.0fs)", epoch + 1, step, total, loss.item(),
                         ema, time.time() - started)
    model.eval()

    if args.calibration:
        cal_rows, cal_states = fitting(engine, *question_rows(load_records(args.calibration)))
        config.temperature = fit_temperature(collect_logits(engine, cal_rows, cal_states), [q["label"] for q in cal_rows])
        log.info("fitted temperature %.3f on %d questions", config.temperature, len(cal_rows))
    config.trained = True
    engine.save(args.out)

    report = {"train_questions": len(rows), "epochs": args.epochs, "seconds": round(time.time() - started),
              "temperature": config.temperature}
    if args.test:
        report["test"] = evaluate(engine, load_records(args.test))
    with open(f"{args.out}/metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--calibration", help="held-out JSONL for temperature scaling")
    ap.add_argument("--test", help="JSONL to evaluate after training")
    ap.add_argument("--out", default="runs/dragonfly-s")
    ap.add_argument("--tier", choices=["S", "M"], default="S")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base",
                    help="S: an encoder such as answerdotai/ModernBERT-base; M: a decoder such as Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=1536,
                    help="S: tokens per question row; M: tokens per question branch")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-checkpointing", action="store_true", help="less GPU memory, ~30%% slower")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, help="backbone / LoRA learning rate (default 3e-5 for S, 2e-4 for M)")
    ap.add_argument("--no-head-norm", action="store_true", help="old head without LayerNorm (only to reproduce earlier runs)")
    ap.add_argument("--head-lr", type=float, help="pointer head learning rate (default 1e-3 for S, 3e-4 for M)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--log-every", type=int, default=50)
    print(json.dumps(train(ap.parse_args()), indent=2))


if __name__ == "__main__":
    main()
