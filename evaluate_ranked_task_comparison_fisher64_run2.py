#!/usr/bin/env python3
"""Full-validation evaluation of saved naive and EWC adapters."""
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 256

def load_rows(name):
    with (ROOT / "data/continual_tasks" / name).open() as f:
        return [json.loads(line) for line in f if line.strip()]

def encode(tokenizer, row):
    prompt = tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    answer = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer = answer + [tokenizer.eos_token_id]
    if len(answer) >= MAX_LENGTH:
        answer = answer[-(MAX_LENGTH - 1):]
        prompt = []
    else:
        prompt = prompt[-(MAX_LENGTH - len(answer)):]
    ids = prompt + answer
    labels = [-100] * len(prompt) + answer
    return ids, labels

@torch.no_grad()
def evaluate(model, rows, tokenizer, device):
    model.eval()
    total_nll, total_tokens, example_losses = 0.0, 0, []
    for row in rows:
        ids, labels = encode(tokenizer, row)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        y = torch.tensor([labels], dtype=torch.long, device=device)
        out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
        count = int((y != -100).sum().item())
        if count and torch.isfinite(out.loss):
            total_nll += float(out.loss.float().cpu()) * count
            total_tokens += count
            example_losses.append(float(out.loss.float().cpu()))
    return {
        "examples_evaluated": len(example_losses),
        "token_weighted_loss": total_nll / total_tokens if total_tokens else None,
        "mean_example_loss": sum(example_losses) / len(example_losses) if example_losses else None,
        "supervised_tokens": total_tokens,
    }

def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    vals = {
        "task_A": load_rows("task_a_val.jsonl"),
        "task_B": load_rows("task_b_val.jsonl"),
    }
    results = {
        "note": "Full-validation loss; ranked historical SFT proxy, not online RL or MACL evidence.",
        "device": str(device),
        "results": {},
    }

    for arm in ("naive", "ewc"):
        results["results"][arm] = {}
        for stage, adapter_name in (
            ("after_task_A", "after_task_A_adapter"),
            ("after_task_B_task_A_prime", "after_task_B_adapter"),
        ):
            adapter_path = ROOT / "outputs" / f"ranked_task_comparison_fisher64_run2_{arm}" / adapter_name
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            base_model.to(device)
            model = PeftModel.from_pretrained(base_model, adapter_path)
            scores = {
                task: evaluate(model, rows, tokenizer, device)
                for task, rows in vals.items()
            }
            results["results"][arm][stage] = scores
            print(f"{arm} | {stage} | {json.dumps(scores)}", flush=True)
            del model, base_model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    out = ROOT / "outputs" / "ranked_task_comparison_fisher64_run2_full_validation.json"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {out}")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved full-validation report: {out}")

if __name__ == "__main__":
    main()
