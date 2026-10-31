import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_allocation_format_test import (
    MODEL_NAME, VAL_FILE, read_valid_allocations,
    COUNTS_RE, PARTNER_COUNTS_RE,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "data_source/end-to-end-negotiator/src/data/negotiate/val.txt"
OUTPUT_FILE = ROOT / "outputs/macl_v5_matched_two_agent_rewards.csv"
GAP_WEIGHT = 1.0

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


def extract(text, tag):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text)
    if not match:
        raise ValueError(f"Missing <{tag}> in source record")
    return match.group(1).strip()


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable.")
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_FILE}")

    raw_lines = [
        line for line in SOURCE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with VAL_FILE.open(encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]

    if len(raw_lines) != len(all_rows):
        raise RuntimeError("Source and SFT validation record counts differ.")

    # Reuse the same first 32 valid prompts used in the earlier evaluation.
    valid_rows = read_valid_allocations(VAL_FILE)[:32]
    valid_prompts = {row["prompt"] for row in valid_rows}
    if len(valid_prompts) != len(valid_rows):
        raise RuntimeError("Duplicate prompts in evaluation subset.")

    paired_rows = []
    for index, (raw, row) in enumerate(zip(raw_lines, all_rows), start=1):
        if row.get("prompt", "").strip() not in valid_prompts:
            continue

        own_input = list(map(int, extract(raw, "input").split()))
        partner_input = list(map(int, extract(raw, "partner_input").split()))
        if len(own_input) != 6 or len(partner_input) != 6:
            raise ValueError(f"Unexpected input shape on source row {index}")

        counts = own_input[0::2]
        own_values = own_input[1::2]
        partner_counts = partner_input[0::2]
        partner_values = partner_input[1::2]

        if counts != partner_counts:
            raise ValueError(f"Stock mismatch on source row {index}")

        # Confirm prompt counts and source counts agree.
        prompt = row["prompt"]
        own_match = COUNTS_RE.search(prompt)
        partner_match = PARTNER_COUNTS_RE.search(prompt)
        if not own_match or not partner_match:
            raise ValueError(f"Missing prompt counts on source row {index}")
        if list(map(int, own_match.groups())) != counts:
            raise ValueError(f"Own stock counts mismatch on source row {index}")
        if list(map(int, partner_match.groups())) != counts:
            raise ValueError(f"Partner stock counts mismatch on source row {index}")

        paired_rows.append({
            "prompt": prompt,
            "counts": counts,
            "own_values": own_values,
            "partner_values": partner_values,
        })

    if len(paired_rows) != 32:
        raise RuntimeError(
            f"Expected 32 aligned evaluation rows, got {len(paired_rows)}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    jobs = [
        (method, seed,
         ROOT / f"outputs/macl_sft_{method}_v5_matched_s{seed}/agent0_adapter")
        for seed in (17, 29, 41)
        for method in ("naive", "ewc", "macl")
    ]
    results = []

    for method, seed, adapter_dir in jobs:
        print(f"\n=== {method.upper()} seed={seed} ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        model.to("mps")
        model.eval()

        valid = 0
        invalid = 0
        own_payoffs = []
        partner_payoffs = []
        self_interest_rewards = []
        own_fairness_rewards = []
        partner_fairness_rewards = []
        payoff_gaps = []

        for i, row in enumerate(paired_rows, start=1):
            chat = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat, return_tensors="pt").to("mps")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            match = FORMAT.fullmatch(generated)
            if not match:
                invalid += 1
                continue

            quantities = list(map(int, match.groups()))
            own_allocation = quantities[:3]
            partner_allocation = quantities[3:]
            counts = row["counts"]

            if not all(
                own_allocation[j] + partner_allocation[j] == counts[j]
                for j in range(3)
            ):
                invalid += 1
                continue

            own_pay = sum(
                v * q for v, q in zip(row["own_values"], own_allocation)
            )
            partner_pay = sum(
                v * q for v, q in zip(row["partner_values"], partner_allocation)
            )
            gap = abs(own_pay - partner_pay)

            valid += 1
            own_payoffs.append(own_pay)
            partner_payoffs.append(partner_pay)
            self_interest_rewards.append(own_pay)
            own_fairness_rewards.append(own_pay - GAP_WEIGHT * gap)
            partner_fairness_rewards.append(partner_pay - GAP_WEIGHT * gap)
            payoff_gaps.append(gap)

        n = len(paired_rows)
        results.append({
            "method": method,
            "seed": seed,
            "examples": n,
            "valid_allocations": valid,
            "invalid_allocations": invalid,
            "valid_rate": valid / n,
            "mean_own_payoff_valid_only": (
                sum(own_payoffs) / valid if valid else ""
            ),
            "mean_partner_payoff_valid_only": (
                sum(partner_payoffs) / valid if valid else ""
            ),
            "mean_total_payoff_valid_only": (
                (sum(own_payoffs) + sum(partner_payoffs)) / valid if valid else ""
            ),
            "mean_payoff_gap_valid_only": (
                sum(payoff_gaps) / valid if valid else ""
            ),
            "mean_own_fairness_reward_valid_only": (
                sum(own_fairness_rewards) / valid if valid else ""
            ),
            "mean_partner_fairness_reward_valid_only": (
                sum(partner_fairness_rewards) / valid if valid else ""
            ),
            "gap_weight": GAP_WEIGHT,
            "evaluation": "offline historical-scenario proxy; valid predictions only",
        })

        print(
            f"RESULT {method} seed={seed}: valid={valid}/{n}, "
            f"own payoff={results[-1]['mean_own_payoff_valid_only']}, "
            f"partner payoff={results[-1]['mean_partner_payoff_valid_only']}, "
            f"own fairness reward={results[-1]['mean_own_fairness_reward_valid_only']}",
            flush=True,
        )

        del model, base
        torch.mps.empty_cache()

    with OUTPUT_FILE.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved summary: {OUTPUT_FILE}")
    print(
        "Important: rewards are computed only for valid predictions. "
        "Fairness reward is an illustrative design choice, not a verified "
        "transcription of the MACL paper."
    )


if __name__ == "__main__":
    main()
