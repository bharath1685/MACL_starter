import csv
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_allocation_format_test import (
    MODEL_NAME, VAL_FILE, read_valid_allocations, COUNTS_RE
)

SYSTEM_PROMPT = (
    "You are an allocation agent. Output exactly six space-separated "
    "fields in this order: item0=N item1=N item2=N item0=N item1=N item2=N. "
    "Use non-negative integers only. Output no dialogue, explanation, "
    "or extra text."
)
EXACT_FORMAT = re.compile(
    r"item0=(\d+) item1=(\d+) item2=(\d+) "
    r"item0=(\d+) item1=(\d+) item2=(\d+)"
)

def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable.")

    device = "mps"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rows = read_valid_allocations(VAL_FILE)[:32]
    if not rows:
        raise RuntimeError("No valid validation examples found.")

    jobs = [
        (method, seed,
         f"outputs/macl_sft_{method}_v5_matched_s{seed}/agent0_adapter")
        for seed in (17, 29, 41)
        for method in ("naive", "ewc", "macl")
    ]
    results = []

    for method, seed, adapter_dir in jobs:
        print(f"\n=== {method.upper()} seed={seed} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.to(device)
        model.eval()

        format_ok = 0
        counts_ok = 0
        for i, row in enumerate(rows, start=1):
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            match = EXACT_FORMAT.fullmatch(generated)
            own_match = COUNTS_RE.search(row["prompt"])
            if not own_match:
                raise ValueError(f"Cannot read stock counts in example {i}.")
            available = list(map(int, own_match.groups()))
            valid_format = match is not None
            valid_counts = False
            if valid_format:
                values = list(map(int, match.groups()))
                valid_counts = all(
                    values[j] + values[j + 3] == available[j]
                    for j in range(3)
                )
            format_ok += int(valid_format)
            counts_ok += int(valid_counts)

            print(
                f"{i:02d}/32 format={valid_format} "
                f"counts={valid_counts}",
                flush=True,
            )

        results.append({
            "method": method,
            "seed": seed,
            "adapter": adapter_dir,
            "examples": len(rows),
            "format_valid": format_ok,
            "count_consistent": counts_ok,
        })
        del model, base
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(
            f"RESULT {method} seed={seed}: "
            f"format={format_ok}/{len(rows)}, "
            f"count-consistent={counts_ok}/{len(rows)}",
            flush=True,
        )

    output_path = "outputs/macl_v5_matched_behavior_summary.csv"
    with open(output_path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved summary: {output_path}")
    print("Diagnostic only: these metrics are not negotiation rewards.")

if __name__ == "__main__":
    main()
