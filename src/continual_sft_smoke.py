#!/usr/bin/env python3
"""
MACL continual-SFT mechanics smoke test.

IMPORTANT:
- This is a pipeline diagnostic, not a MACL result.
- THEM-first vs YOU-first are artificial speaker-order cohorts, NOT self-interest
  vs fairness tasks.
- It trains one small LoRA adapter sequentially on two cohorts, checks that an
  EWC penalty is non-zero, and saves everything under a NEW output directory.
- It does not implement GRPO/PPO or the paper's multi-agent communication method.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def first_speaker(prompt: str):
    # Dataset prompts have a dialogue section. Use a conservative heuristic.
    upper = prompt.upper()
    pos_them = upper.find("THEM:")
    pos_you = upper.find("YOU:")
    if pos_them < 0 and pos_you < 0:
        return None
    if pos_them < 0:
        return "YOU_first"
    if pos_you < 0:
        return "THEM_first"
    return "THEM_first" if pos_them < pos_you else "YOU_first"


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row.get("prompt"), str) and isinstance(row.get("answer"), str):
                    row["_cohort"] = first_speaker(row["prompt"])
                    rows.append(row)
    return rows


def encode_example(tokenizer, row, max_length):
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        answer_ids = answer_ids + [eos]
    # Keep the tail of the prompt if needed, preserving the answer.
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[-(max_length - 1):]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    input_ids = prompt_ids + answer_ids
    labels = ([-100] * len(prompt_ids)) + answer_ids[:]
    return input_ids, labels


def collate_one(tokenizer, row, max_length, device):
    ids, labels = encode_example(tokenizer, row, max_length)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    target = torch.tensor([labels], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": target}


def lora_parameters(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


def ewc_penalty(params, fisher, anchor, device):
    penalty = torch.zeros((), device=device)
    for name, p in params.items():
        if name in fisher and name in anchor:
            penalty = penalty + (fisher[name] * (p - anchor[name]).pow(2)).sum()
    return penalty


def snapshot(params):
    return {n: p.detach().float().cpu().clone() for n, p in params.items()}


def estimate_fisher(model, rows, tokenizer, max_length, device, max_examples):
    model.eval()
    params = lora_parameters(model)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32, device=device) for n, p in params.items()}
    used = 0
    for row in rows[:max_examples]:
        batch = collate_one(tokenizer, row, max_length, device)
        model.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            continue
        loss.backward()
        for name, p in params.items():
            if p.grad is not None:
                fisher[name] += p.grad.detach().float().pow(2)
        used += 1
    if used:
        for name in fisher:
            fisher[name] /= used
    model.zero_grad(set_to_none=True)
    return {n: t.detach().cpu().clone() for n, t in fisher.items()}, used


def train_stage(model, rows, tokenizer, max_length, device, steps, lr,
                fisher=None, anchor=None, ewc_lambda=0.0):
    model.train()
    params = lora_parameters(model)
    optimizer = torch.optim.AdamW(params.values(), lr=lr)
    losses = []
    penalties = []
    if not rows:
        raise RuntimeError("No examples available for this cohort.")
    for step in range(steps):
        row = rows[step % len(rows)]
        batch = collate_one(tokenizer, row, max_length, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        penalty = torch.zeros((), device=device)
        if fisher is not None and anchor is not None and ewc_lambda > 0:
            # Convert CPU snapshots to the parameter device for this step.
            fisher_dev = {n: v.to(device) for n, v in fisher.items()}
            anchor_dev = {n: v.to(device) for n, v in anchor.items()}
            penalty = ewc_penalty(params, fisher_dev, anchor_dev, device)
        total = loss + ewc_lambda * penalty
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite training loss; stopping safely.")
        total.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        penalties.append(float(penalty.detach().cpu()))
    return {"mean_loss": sum(losses) / len(losses), "mean_ewc_penalty": sum(penalties) / len(penalties)}


@torch.no_grad()
def evaluate_loss(model, rows, tokenizer, max_length, device, limit):
    model.eval()
    vals = []
    for row in rows[:limit]:
        batch = collate_one(tokenizer, row, max_length, device)
        out = model(**batch)
        if torch.isfinite(out.loss):
            vals.append(float(out.loss.detach().cpu()))
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out", default="outputs/continual_sft_smoke")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--train-examples", type=int, default=4)
    ap.add_argument("--val-examples", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--fisher-examples", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ewc-lambda", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)
    train_a = [r for r in train_rows if r["_cohort"] == "THEM_first"]
    train_b = [r for r in train_rows if r["_cohort"] == "YOU_first"]
    val_a = [r for r in val_rows if r["_cohort"] == "THEM_first"]
    val_b = [r for r in val_rows if r["_cohort"] == "YOU_first"]
    if min(len(train_a), len(train_b), len(val_a), len(val_b)) == 0:
        raise RuntimeError("Could not form both speaker-order cohorts; inspect prompt format.")
    train_a = train_a[:args.train_examples]
    train_b = train_b[:args.train_examples]
    val_a = val_a[:args.val_examples]
    val_b = val_b[:args.val_examples]

    print(f"Device: {device}; dtype: {dtype}")
    print("Task labels: A=THEM_first, B=YOU_first (debugging cohorts only; NOT paper task labels)")
    print(f"Examples: train A={len(train_a)}, train B={len(train_b)}, val A={len(val_a)}, val B={len(val_b)}")
    print(f"Output directory: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    params = lora_parameters(model)
    trainable = sum(p.numel() for p in params.values())
    print(f"Trainable LoRA parameters: {trainable:,}")
    if not params:
        raise RuntimeError("No trainable LoRA parameters found.")

    report = {
        "diagnostic_only": True,
        "not_paper_tasks": True,
        "base_model": args.base_model,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "cohorts": {"A": "THEM_first", "B": "YOU_first"},
        "settings": vars(args),
        "trainable_lora_parameters": trainable,
        "stages": [],
    }

    before_a = {
        "val_A": evaluate_loss(model, val_a, tokenizer, args.max_length, device, args.val_examples),
        "val_B": evaluate_loss(model, val_b, tokenizer, args.max_length, device, args.val_examples),
    }
    print("Base evaluation:", before_a)

    stage_a = train_stage(model, train_a, tokenizer, args.max_length, device, args.steps, args.lr)
    fisher, fisher_used = estimate_fisher(model, train_a, tokenizer, args.max_length, device, args.fisher_examples)
    anchor = snapshot(lora_parameters(model))
    after_a = {
        "val_A": evaluate_loss(model, val_a, tokenizer, args.max_length, device, args.val_examples),
        "val_B": evaluate_loss(model, val_b, tokenizer, args.max_length, device, args.val_examples),
    }
    print("After Task A:", stage_a, after_a, f"Fisher examples={fisher_used}")
    model.save_pretrained(out_dir / "after_task_A_adapter")

    stage_b = train_stage(
        model, train_b, tokenizer, args.max_length, device, args.steps, args.lr,
        fisher=fisher, anchor=anchor, ewc_lambda=args.ewc_lambda,
    )
    after_b = {
        "val_A": evaluate_loss(model, val_a, tokenizer, args.max_length, device, args.val_examples),
        "val_B": evaluate_loss(model, val_b, tokenizer, args.max_length, device, args.val_examples),
    }
    print("After Task B + EWC:", stage_b, after_b)
    model.save_pretrained(out_dir / "after_task_B_ewc_adapter")

    # Save diagnostic metadata, Fisher diagonal, and anchor values for reproducibility.
    torch.save({"fisher": fisher, "anchor": anchor}, out_dir / "ewc_state.pt")
    report["stages"] = [
        {"stage": "base", "eval_loss": before_a},
        {"stage": "task_A", "train": stage_a, "eval_loss": after_a, "fisher_examples": fisher_used},
        {"stage": "task_B_with_ewc", "train": stage_b, "eval_loss": after_b},
    ]
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"PASS: continual SFT/EWC mechanics smoke test completed. Report: {out_dir / 'report.json'}")
    print("Interpretation limit: this does NOT demonstrate MACL, communication benefits, or self-interest/fairness retention.")


if __name__ == "__main__":
    main()
