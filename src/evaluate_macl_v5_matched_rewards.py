import csv
import re
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_allocation_format_test import (
    MODEL_NAME, VAL_FILE, read_valid_allocations, COUNTS_RE,
    PARTNER_COUNTS_RE,
)

SYSTEM_PROMPT = (
    "You are an allocation agent. Output exactly six space-separated "
    "fields in this order: item0=N item1=N item2=N item0=N item1=N item2=N. "
    "Use non-negative integers only. Output no dialogue, explanation, "
    "or extra text."
)
FORMAT = re.compile(
    r"item0=(\d+) item1=(\d+) item2=(\d+) "
    r"item0=(\d+) item1=(\d+) item2=(\d+)"
)
VALUES_RE = re.compile(
    r"Your private values: \[(\d+), (\d+), (\d+)\]"
)
GAP_WEIGHT = 1.0


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable.")

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
        model.to("mps")
        model.eval()

        valid = invalid = 0
        own_payoffs = []
        partner_payoffs = []
        fair_scores = []
        gaps = []

        for i, row in enumerate(rows, start=1):
            prompt = row["prompt"]
            counts_match = COUNTS_RE.search(prompt)
            partner_counts_match = PARTNER_COUNTS_RE.search(prompt)
            values_match = VALUES_RE.search(prompt)
            if not counts_match or not partner_counts_match or not values_match:
                raise ValueError(f"Missing scenario fields in validation row {i}.")

            counts = list(map(int, counts_match.groups()))
            partner_counts = list(map(int, partner_counts_match.groups()))
            own_values = list(map(int, values_match.groups()))

            # The other agent's private values are not in this prompt.
            # Do not infer them from the dialogue or recorded answer.
            chat = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat, return_tensors="pt").to("mps")
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=40, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            match = FORMAT.fullmatch(generated)

            allocation_valid = (
                match is not None
                and counts == partner_counts
            )
            if allocation_valid:
                q_own = list(map(int, match.groups()[:3]))
                q_partner = list(map(int, match.groups()[3:]))
                allocation_valid = all(
                    q_own[j] + q_partner[j] == counts[j]
                    for j in range(3)
                )
            if not allocation_valid:
                invalid += 1
                print(f"{i:02d}/32 invalid allocation", flush=True)
                continue

            # Only own payoff is identifiable from this prompt's private values.
            own_pay = sum(v * q for v, q in zip(own_values, q_own))
            valid += 1
            own_payoffs.append(own_pay)
            print(f"{i:02d}/32 valid allocation; own payoff={own_pay}", flush=True)

        results.append({
            "method": method,
            "seed": seed,
            "examples": len(rows),
            "valid_allocations": valid,
            "invalid_allocations": invalid,
            "valid_rate": valid / len(rows),
            "mean_own_payoff_valid_only": (
                sum(own_payoffs) / len(own_payoffs) if own_payoffs else ""
            ),
            "fairness_adjusted_reward": "NOT_COMPUTED",
            "note": (
                "Partner private values are absent from prompt; "
                "fairness reward cannot be computed without them."
            ),
        })

        del model, base
        torch.mps.empty_cache()
        print(
            f"RESULT {method} seed={seed}: valid={valid}/{len(rows)}, "
            f"invalid={invalid}/{len(rows)}, "
            f"mean own payoff={results[-1]['mean_own_payoff_valid_only']}",
            flush=True,
        )

    output_path = Path("outputs/macl_v5_matched_rewards_summary.csv")
    with output_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved summary: {output_path}")
    print("Own payoff is a historical-scenario proxy, not interactive negotiation reward.")
    print("Fairness-adjusted reward was not computed because partner values are hidden.")


if __name__ == "__main__":
    main()
