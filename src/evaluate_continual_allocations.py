import gc
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = ROOT / "outputs" / "continual_allocation_evaluation.json"
TASKS = ROOT / "data" / "continual_tasks"
ARMS = ("naive", "ewc")
STAGES = {
    "after_A": "after_task_A_adapter",
    "after_B": "after_task_B_adapter",
    "after_A_prime": "after_task_A_prime_adapter",
}
ITEM_RE = re.compile(r"item([0-2])=(\d+)")
COUNTS_RE = re.compile(r"Your item counts: \[(\d+), (\d+), (\d+)\]")

def read_rows(path, limit=32):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()][:limit]

def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    results = {
        "warning": "Validation-only diagnostic; not interactive MARL or proof of MACL effectiveness.",
        "examples_per_task": 32,
        "metrics": {},
    }

    rows_by_task = {
        "task_A": read_rows(TASKS / "task_a_val.jsonl"),
        "task_B": read_rows(TASKS / "task_b_val.jsonl"),
    }

    for arm in ARMS:
        for stage, adapter_name in STAGES.items():
            adapter = ROOT / "outputs" / f"ranked_task_comparison_aba_full211_{arm}" / adapter_name
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing adapter: {adapter}")

            print(f"\nLoading {arm} / {stage}", flush=True)
            base = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype)
            model = PeftModel.from_pretrained(base, adapter).to(device)
            model.eval()
            key = f"{arm}/{stage}"
            results["metrics"][key] = {}

            for task, rows in rows_by_task.items():
                counts = {"examples": len(rows), "format_valid": 0,
                          "stock_consistent": 0, "exact_target": 0}
                for i, row in enumerate(rows, 1):
                    inputs = tokenizer(
                        row["prompt"], return_tensors="pt",
                        add_special_tokens=True
                    ).to(device)
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs, max_new_tokens=64, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    answer = tokenizer.decode(
                        generated[0, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    found = ITEM_RE.findall(answer)
                    valid = (len(found) == 6 and
                             [int(x[0]) for x in found] == [0, 1, 2, 0, 1, 2])
                    counts["format_valid"] += int(valid)
                    match = COUNTS_RE.search(row["prompt"])
                    feasible = False
                    if valid and match:
                        available = list(map(int, match.groups()))
                        values = [int(x[1]) for x in found]
                        feasible = all(
                            values[j] + values[j + 3] <= available[j]
                            for j in range(3)
                        )
                    counts["stock_consistent"] += int(feasible)
                    target = ITEM_RE.findall(row["answer"])
                    counts["exact_target"] += int(found == target and len(target) == 6)

                results["metrics"][key][task] = counts
                print(f"{key} | {task}: {counts}", flush=True)

            del model, base
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved report: {OUT}", flush=True)

if __name__ == "__main__":
    main()
