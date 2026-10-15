"""B9: one LoRA fine-tune on Transformers and PEFT with Token Factory hyperparameters.

Runs on a MacBook (MPS), Kaggle (CUDA) or CPU. Reads train and validation JSONL in the
Token Factory chat format, puts the loss on the final assistant answer only, and
writes to --out:
  metrics.jsonl  a line per optimizer step (train_*) and per evaluation (valid_*)
  summary.json   hyperparameters, fault, seed, device, tokens, time, final metrics
  adapter/       the LoRA adapter: adapter_model.safetensors and adapter_config.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench.faults.recipe import HEALTHY_HP, STRENGTHS, inject

# Batch width is rounded up to this: every new width grows the MPS cache
PAD_MULTIPLE = 16


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tok: Any, messages: list[dict[str, str]], context_length: int) -> dict:
    """Token ids of the whole chat; labels keep only the final assistant message."""
    full = tok.apply_chat_template(messages, tokenize=False)
    # enable_thinking=False makes Qwen3 render the same empty think block
    # as in the full chat; other templates ignore the flag
    prompt = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("the chat template renders the prompt unlike the full chat")
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
    return {"input_ids": ids[:context_length], "labels": labels[:context_length]}


def collate(rows: list[dict], pad_id: int, device: str) -> dict[str, torch.Tensor]:
    width = max(len(r["input_ids"]) for r in rows)
    width = math.ceil(width / PAD_MULTIPLE) * PAD_MULTIPLE
    ids, labels, mask = [], [], []
    for r in rows:
        pad = width - len(r["input_ids"])
        ids.append(r["input_ids"] + [pad_id] * pad)
        labels.append(r["labels"] + [-100] * pad)
        mask.append([1] * len(r["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(mask, device=device),
    }


def free_cache(device: str) -> None:
    # without this the MPS cache fills 16 GB of unified memory within 20 steps
    if device == "mps":
        torch.mps.empty_cache()


def loss_and_accuracy(model: Any, batch: dict) -> tuple[torch.Tensor, int, int]:
    """Mean loss over answer tokens, the number of correct and of answer tokens."""
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    ).logits
    logits, labels = logits[:, :-1].float(), batch["labels"][:, 1:]
    keep = labels != -100
    loss = F.cross_entropy(logits[keep], labels[keep])
    correct = int((logits[keep].argmax(-1) == labels[keep]).sum())
    return loss, correct, int(keep.sum())


@torch.no_grad()
def evaluate(model: Any, rows: list[dict], pad_id: int, device: str, bs: int) -> dict:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for i in range(0, len(rows), bs):
        batch = collate(rows[i : i + bs], pad_id, device)
        loss, ok, n = loss_and_accuracy(model, batch)
        loss_sum, correct, total = loss_sum + loss.item() * n, correct + ok, total + n
    free_cache(device)
    model.train()
    return {
        "valid_loss": loss_sum / total,
        "valid_mean_token_accuracy": correct / total,
    }


@torch.no_grad()
def exact_match(model: Any, tok: Any, chats: list[dict], device: str) -> float:
    """Share of greedy answers equal to the reference. Stand-in until core/eval (A13)."""
    model.eval()
    hits = 0
    for chat in chats:
        prompt = tok.apply_chat_template(
            chat["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=32, do_sample=False)
        answer = tok.decode(
            out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        hits += answer.strip() == chat["messages"][-1]["content"].strip()
    free_cache(device)
    model.train()
    return hits / len(chats)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lr_factor(step: int, total: int, warmup: int, schedule: str) -> float:
    """Linear warmup, then constant or linear decay to zero.

    Token Factory decays the rate over the run (docs/b6-post-training.md), so linear
    is the default; whether TF decays linearly or by cosine is not known yet.
    """
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    return max(0.0, (total - step) / max(1, total - warmup))


def train(
    train_path: str | Path,
    valid_path: str | Path,
    out: str | Path,
    model_id: str = "Qwen/Qwen3-0.6B",
    hp: dict[str, Any] | None = None,
    fault: str | None = None,
    strength: float | None = None,
    seed: int = 0,
    lr_schedule: str = "linear",
    evals_per_epoch: int = 5,
    generate: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    started = time.monotonic()
    hp = {**HEALTHY_HP, **(hp or {})}
    if fault:
        hp = inject(hp, fault, strength)
    device = pick_device(device)
    random.seed(seed)
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_id)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    train_chats, valid_chats = read_jsonl(train_path), read_jsonl(valid_path)
    ctx = hp["context_length"]
    train_rows = [encode(tok, c["messages"], ctx) for c in train_chats]
    valid_rows = [encode(tok, c["messages"], ctx) for c in valid_chats]

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    lora = LoraConfig(
        r=hp["lora_r"],
        lora_alpha=hp["lora_alpha"],
        lora_dropout=hp["lora_dropout"],
        # Token Factory adapters cover all seven linear projections
        # (adapter_config.json of a TF checkpoint, docs/b6-post-training.md)
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora).to(device)
    model.train()

    bs = hp["batch_size"]
    steps_per_epoch = math.ceil(len(train_rows) / bs)
    total_steps = steps_per_epoch * hp["n_epochs"]
    warmup = math.ceil(hp["warmup_ratio"] * total_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total_steps, warmup, lr_schedule)
    )
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")

    def write(row: dict) -> None:
        row["elapsed_s"] = round(time.monotonic() - started, 2)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    write({"step": 0, "epoch": 0.0, **evaluate(model, valid_rows, pad_id, device, bs)})
    order = random.Random(seed)
    step, answer_tokens, input_tokens = 0, 0, 0
    for _ in range(hp["n_epochs"]):
        idx = list(range(len(train_rows)))
        order.shuffle(idx)
        for i in range(0, len(idx), bs):
            batch = collate([train_rows[j] for j in idx[i : i + bs]], pad_id, device)
            loss, correct, n = loss_and_accuracy(model, batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
            lr = sched.get_last_lr()[0]
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            free_cache(device)
            step += 1
            answer_tokens += n
            input_tokens += int(batch["attention_mask"].sum())
            epoch = round(step / steps_per_epoch, 4)
            write(
                {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "train_mean_token_accuracy": correct / n,
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "answer_tokens": answer_tokens,
                }
            )
            if step % eval_every == 0 or step % steps_per_epoch == 0:
                metrics = evaluate(model, valid_rows, pad_id, device, bs)
                write({"step": step, "epoch": epoch, **metrics})

    final = evaluate(model, valid_rows, pad_id, device, bs)
    if generate:
        final["exact_match"] = exact_match(model, tok, valid_chats, device)
    model.save_pretrained(out / "adapter")
    summary = {
        "model": model_id,
        "fault": fault,
        "strength": strength,
        "seed": seed,
        "hyperparameters": hp,
        "lr_schedule": lr_schedule,
        "device": device,
        "train_examples": len(train_rows),
        "valid_examples": len(valid_rows),
        "steps": step,
        "answer_tokens": answer_tokens,
        "input_tokens": input_tokens,
        "wall_seconds": round(time.monotonic() - started, 1),
        "final": final,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--hp", default="{}", help="JSON with hyperparameter overrides")
    parser.add_argument("--fault", choices=sorted(STRENGTHS))
    parser.add_argument("--strength", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr-schedule", choices=["linear", "constant"], default="linear"
    )
    parser.add_argument("--evals-per-epoch", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="exact match on valid")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.fault and args.strength is None:
        parser.error("--fault needs --strength")

    summary = train(
        args.train,
        args.valid,
        args.out,
        model_id=args.model,
        hp=json.loads(args.hp),
        fault=args.fault,
        strength=args.strength,
        seed=args.seed,
        lr_schedule=args.lr_schedule,
        evals_per_epoch=args.evals_per_epoch,
        generate=args.generate,
        device=args.device,
    )
    print(json.dumps({k: summary[k] for k in ("steps", "wall_seconds", "final")}))


if __name__ == "__main__":
    main()
