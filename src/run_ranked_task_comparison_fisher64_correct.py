#!/usr/bin/env python3
"""Controlled sequential SFT vs fixed-EWC comparison on ranked task data.

This is a supervised continual-learning proxy, not online MARL or MACL.
Uses a fresh base model for each arm and refuses to overwrite outputs.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from continual_sft_instrumented import (
    read_jsonl, encode_example, collate_one, lora_parameters,
    ewc_penalty, snapshot, estimate_fisher, train_stage, evaluate_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, rows, tokenizer, max_length, device, limit):
    selected = rows[:limit] if limit else rows
    loss = evaluate_loss(model, selected, tokenizer, max_length, device, len(selected))
    return {"loss": loss, "examples": len(selected)}


def run_arm(arm, args, tokenizer, train_a, train_b, val_a, val_b, device, dtype):
    out = ROOT / "outputs" / f"ranked_task_comparison_fisher64_correct_{arm}"
    out.mkdir(parents=True, exist_ok=False)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    params = lora_parameters(model)
    if not params:
        raise RuntimeError("No trainable LoRA parameters found")

    report = {
        "arm": arm,
        "method": "ranked historical supervised continual-learning proxy",
        "not_online_rl": True,
        "not_macl_reproduction": True,
        "seed": args.seed,
        "settings": vars(args),
        "stages": [],
    }

    def ev(stage):
        result = {
            "task_A": evaluate(model, val_a, tokenizer, args.max_length, device, args.val_examples),
            "task_B": evaluate(model, val_b, tokenizer, args.max_length, device, args.val_examples),
        }
        print(f"[{arm}] {stage}: {result}", flush=True)
        report["stages"].append({"stage": stage, "validation": result})
        return result

    ev("base")
    a_metrics = train_stage(
        model, train_a, tokenizer, args.max_length, device,
        args.steps, args.lr, stage_name="task_A",
        log_path=out / "task_A_step_metrics.json",
    )
    report["task_A_training"] = a_metrics
    model.save_pretrained(out / "after_task_A_adapter")
    fisher = anchor = None

    if arm == "ewc":
        fisher, used = estimate_fisher(
            model, train_a, tokenizer, args.max_length, device, args.fisher_examples
        )
        anchor = snapshot(lora_parameters(model))
        report["fisher_examples_used"] = used
        torch.save({"fisher": fisher, "anchor": anchor}, out / "ewc_state.pt")

    ev("after_task_A")
    b_metrics = train_stage(
        model, train_b, tokenizer, args.max_length, device,
        args.steps, args.lr,
        fisher=fisher, anchor=anchor,
        ewc_lambda=args.ewc_lambda if arm == "ewc" else 0.0,
        stage_name="task_B_with_ewc" if arm == "ewc" else "task_B_naive",
        log_path=out / "task_B_step_metrics.json",
    )
    report["task_B_training"] = b_metrics
    ev("after_task_B_task_A_prime")
    model.save_pretrained(out / "after_task_B_adapter")

    with (out / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[{arm}] Report: {out / 'report.json'}", flush=True)

    del model, params, fisher, anchor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--train-examples", type=int, default=8)
    ap.add_argument("--val-examples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fisher-examples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ewc-lambda", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    task_dir = ROOT / "data" / "continual_tasks"
    train_a = read_jsonl(task_dir / "task_a_train.jsonl")[:args.train_examples]
    train_b = read_jsonl(task_dir / "task_b_train.jsonl")[:args.train_examples]
    val_a = read_jsonl(task_dir / "task_a_val.jsonl")[:args.val_examples]
    val_b = read_jsonl(task_dir / "task_b_val.jsonl")[:args.val_examples]
    if min(map(len, (train_a, train_b, val_a, val_b))) == 0:
        raise RuntimeError("At least one task split is empty")

    if torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    elif torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32

    print(f"Device={device}, dtype={dtype}", flush=True)
    print(
        f"Train A/B={len(train_a)}/{len(train_b)}, "
        f"validation A/B={len(val_a)}/{len(val_b)}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_arm("naive", args, tokenizer, train_a, train_b, val_a, val_b, device, dtype)
    run_arm("ewc", args, tokenizer, train_a, train_b, val_a, val_b, device, dtype)
    print("Both arms completed. Compare the two report.json files.", flush=True)


if __name__ == "__main__":
    main()
