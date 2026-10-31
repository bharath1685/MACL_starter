#!/usr/bin/env python3
"""
Controlled sequential-SFT comparison: naive sequential fine-tuning vs fixed EWC.

This is a diagnostic, not a MACL result:
- Task A/B are artificial speaker-order cohorts (THEM_first / YOU_first).
- It is single-agent SFT, not the paper's self-interest/fairness RL task sequence.
- Same seed, selected examples, initialization, steps, and optimizer settings are
  used for both arms. Separate fresh base models are loaded for each arm.
- A unique output directory is required; nothing existing is overwritten.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def first_speaker(prompt):
    upper = prompt.upper()
    t, y = upper.find("THEM:"), upper.find("YOU:")
    if t < 0 and y < 0:
        return None
    if t < 0:
        return "YOU_first"
    if y < 0:
        return "THEM_first"
    return "THEM_first" if t < y else "YOU_first"


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


def encode(tokenizer, row, max_length):
    p = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    a = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        a.append(tokenizer.eos_token_id)
    if len(a) >= max_length:
        a = a[-(max_length - 1):]
        p = []
    else:
        p = p[-(max_length - len(a)):]
    ids = p + a
    labels = [-100] * len(p) + a[:]
    return ids, labels


def batch_one(tokenizer, row, max_length, device):
    ids, labels = encode(tokenizer, row, max_length)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def trainable_params(model):
    return {k: v for k, v in model.named_parameters() if v.requires_grad}


def cpu_snapshot(params):
    return {k: p.detach().float().cpu().clone() for k, p in params.items()}


def ewc_penalty(params, fisher, anchor, device):
    result = torch.zeros((), device=device)
    for k, p in params.items():
        if k in fisher and k in anchor:
            result = result + (fisher[k] * (p.float() - anchor[k]).square()).sum()
    return result


def estimate_fisher(model, rows, tokenizer, max_length, device, limit):
    model.eval()
    params = trainable_params(model)
    f = {k: torch.zeros_like(p, dtype=torch.float32, device=device) for k, p in params.items()}
    used = 0
    for row in rows[:limit]:
        model.zero_grad(set_to_none=True)
        out = model(**batch_one(tokenizer, row, max_length, device))
        if not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for k, p in params.items():
            if p.grad is not None:
                f[k] += p.grad.detach().float().square()
        used += 1
    model.zero_grad(set_to_none=True)
    if used:
        for k in f:
            f[k] /= used
    return {k: v.detach().cpu().clone() for k, v in f.items()}, used


def eval_loss(model, rows, tokenizer, max_length, device):
    model.eval()
    vals = []
    with torch.no_grad():
        for row in rows:
            out = model(**batch_one(tokenizer, row, max_length, device))
            if torch.isfinite(out.loss):
                vals.append(float(out.loss.detach().cpu()))
    return sum(vals) / len(vals) if vals else None


def run_arm(arm, args, tokenizer, task_a, task_b, val_a, val_b, device, dtype, arm_dir):
    arm_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM"
    ))
    params = trainable_params(model)
    opt = torch.optim.AdamW(params.values(), lr=args.lr)
    logs = []
    evals = {
        "base": {
            "val_A": eval_loss(model, val_a, tokenizer, args.max_length, device),
            "val_B": eval_loss(model, val_b, tokenizer, args.max_length, device),
        }
    }
    fisher, anchor = None, None

    def train_task(rows, stage, with_ewc):
        model.train()
        stage_logs = []
        for step in range(args.steps):
            row = rows[step % len(rows)]
            opt.zero_grad(set_to_none=True)
            out = model(**batch_one(tokenizer, row, args.max_length, device))
            task_loss = out.loss
            penalty = torch.zeros((), device=device)
            if with_ewc:
                penalty = ewc_penalty(
                    params,
                    {k: v.to(device) for k, v in fisher.items()},
                    {k: v.to(device) for k, v in anchor.items()},
                    device,
                )
            weighted = args.ewc_lambda * penalty
            total = task_loss + weighted
            if not torch.isfinite(total):
                raise RuntimeError(f"{arm}/{stage}: non-finite loss")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            opt.step()
            rowlog = {
                "arm": arm, "stage": stage, "step": step + 1,
                "task_loss": float(task_loss.detach().cpu()),
                "unweighted_ewc_penalty": float(penalty.detach().cpu()),
                "weighted_ewc_penalty": float(weighted.detach().cpu()),
                "total_loss": float(total.detach().cpu()),
                "total_gradient_norm_before_clip": float(torch.as_tensor(grad_norm).detach().cpu()),
            }
            logs.append(rowlog)
            stage_logs.append(rowlog)
            print(
                f"[{arm} {stage} {step+1}/{args.steps}] "
                f"task={rowlog['task_loss']:.6f} "
                f"ewc={rowlog['weighted_ewc_penalty']:.3e} "
                f"grad={rowlog['total_gradient_norm_before_clip']:.3e}"
            )
        return {"mean_task_loss": sum(x["task_loss"] for x in stage_logs)/len(stage_logs),
                "mean_weighted_ewc_penalty": sum(x["weighted_ewc_penalty"] for x in stage_logs)/len(stage_logs)}

    stage_a = train_task(task_a, "task_A", False)
    evals["after_A"] = {
        "val_A": eval_loss(model, val_a, tokenizer, args.max_length, device),
        "val_B": eval_loss(model, val_b, tokenizer, args.max_length, device),
    }
    model.save_pretrained(arm_dir / "after_task_A_adapter")

    if arm == "fixed_ewc":
        fisher, used = estimate_fisher(
            model, task_a, tokenizer, args.max_length, device, args.fisher_examples
        )
        anchor = cpu_snapshot(params)
        torch.save({"fisher": fisher, "anchor": anchor}, arm_dir / "ewc_state.pt")
        print(f"[{arm}] Fisher estimated from {used} examples.")
    else:
        used = 0

    stage_b = train_task(task_b, "task_B", arm == "fixed_ewc")
    evals["after_B"] = {
        "val_A": eval_loss(model, val_a, tokenizer, args.max_length, device),
        "val_B": eval_loss(model, val_b, tokenizer, args.max_length, device),
    }
    model.save_pretrained(arm_dir / "after_task_B_adapter")

    summary = {
        "arm": arm, "trainable_parameters": sum(p.numel() for p in params.values()),
        "fisher_examples": used, "stages": {"task_A": stage_a, "task_B": stage_b},
        "evaluation": evals, "step_logs": logs,
    }
    with open(arm_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    del model, opt, params
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/sft/train.jsonl")
    ap.add_argument("--val", default="data/sft/val.jsonl")
    ap.add_argument("--out", default="outputs/ewc_controlled_comparison")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--train-examples", type=int, default=4)
    ap.add_argument("--val-examples", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--fisher-examples", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ewc-lambda", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.steps < 1 or args.train_examples < 1 or args.val_examples < 1:
        raise ValueError("steps, train-examples, and val-examples must be positive.")

    if torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    elif torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32

    train = read_jsonl(args.train)
    val = read_jsonl(args.val)
    tr_a = [r for r in train if r["_cohort"] == "THEM_first"][:args.train_examples]
    tr_b = [r for r in train if r["_cohort"] == "YOU_first"][:args.train_examples]
    va_a = [r for r in val if r["_cohort"] == "THEM_first"][:args.val_examples]
    va_b = [r for r in val if r["_cohort"] == "YOU_first"][:args.val_examples]
    if min(map(len, (tr_a, tr_b, va_a, va_b))) == 0:
        raise RuntimeError("Unable to form both cohorts from train/validation files.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Device={device}, dtype={dtype}; seed={args.seed}")
    print("CAUTION: cohorts are speaker-order diagnostics, NOT paper task labels.")
    print(f"Train A/B={len(tr_a)}/{len(tr_b)}; validation A/B={len(va_a)}/{len(va_b)}")
    common = dict(args=args, tokenizer=tokenizer, task_a=tr_a, task_b=tr_b,
                  val_a=va_a, val_b=va_b, device=device, dtype=dtype)
    naive = run_arm("naive", **common, arm_dir=out / "naive")
    ewc = run_arm("fixed_ewc", **common, arm_dir=out / "fixed_ewc")

    def retention(summary):
        before = summary["evaluation"]["after_A"]["val_A"]
        after = summary["evaluation"]["after_B"]["val_A"]
        return {"task_A_loss_after_A": before, "task_A_loss_after_B": after,
                "task_A_loss_increase": after - before,
                "task_B_loss_after_B": summary["evaluation"]["after_B"]["val_B"]}

    comparison = {
        "diagnostic_only": True,
        "not_paper_tasks": True,
        "settings": vars(args),
        "device": str(device), "dtype": str(dtype),
        "naive": retention(naive), "fixed_ewc": retention(ewc),
        "interpretation": (
            "Tiny smoke test only. Lower Task A loss increase is a preliminary retention signal, "
            "not proof of statistically meaningful forgetting mitigation. Cohorts are not paper tasks."
        ),
    }
    with open(out / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("\nCOMPARISON")
    print(json.dumps(comparison, indent=2))
    print(f"\nSaved comparison to {out / 'comparison.json'}")


if __name__ == "__main__":
    main()
