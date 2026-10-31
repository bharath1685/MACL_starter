import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_allocation_format_test import (
    MODEL_NAME,
    OUTPUT_DIR,
    VAL_FILE,
    read_valid_allocations,
    COUNTS_RE,
)

SYSTEM_PROMPT = (
    "You are an allocation agent. Output exactly six space-separated "
    "fields in this order: item0=N item1=N item2=N item0=N item1=N item2=N. "
    "Use non-negative integers only. Output no dialogue or explanation."
)

STRICT_FORMAT = re.compile(
    r"item0=(\d+) item1=(\d+) item2=(\d+) "
    r"item0=(\d+) item1=(\d+) item2=(\d+)"
)


def repair_allocation(values, available):
    """Preserve the first proposal where feasible; assign the remainder."""
    first = [
        min(max(int(values[i]), 0), available[i])
        for i in range(3)
    ]
    second = [
        available[i] - first[i]
        for i in range(3)
    ]
    return first, second


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable.")

    adapter_file = OUTPUT_DIR / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Adapter not found: {adapter_file}")

    rows = read_valid_allocations(VAL_FILE)[:32]
    if not rows:
        raise RuntimeError("No eligible validation examples found.")

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.to("mps")
    model.eval()

    raw_format = 0
    raw_feasible = 0
    repaired_feasible = 0
    repair_coverage = 0

    print(f"\nSaved-adapter evaluation: {len(rows)} examples")
    print("No training will be performed.")
    print("Repair is post-processing, not model output.\n")

    for index, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")

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

        match = STRICT_FORMAT.fullmatch(generated)
        counts_match = COUNTS_RE.search(row["prompt"])
        if not counts_match:
            raise ValueError(f"Could not extract item counts in example {index}")

        available = list(map(int, counts_match.groups()))
        format_ok = match is not None
        raw_format += int(format_ok)

        print(f"{index:02d}. Available supply: {available}")
        print(f"    Raw output: {generated!r}")
        print(f"    Raw format valid: {format_ok}")

        if not format_ok:
            print("    Repair: skipped (malformed output)")
            continue

        values = list(map(int, match.groups()))
        first_raw, second_raw = values[:3], values[3:]
        raw_ok = all(
            first_raw[i] + second_raw[i] == available[i]
            for i in range(3)
        )
        raw_feasible += int(raw_ok)

        first_fixed, second_fixed = repair_allocation(values, available)
        fixed_ok = all(
            first_fixed[i] + second_fixed[i] == available[i]
            and first_fixed[i] >= 0
            and second_fixed[i] >= 0
            for i in range(3)
        )
        repaired_feasible += int(fixed_ok)
        repair_coverage += 1

        print(f"    Raw combined supply: "
              f"{[first_raw[i] + second_raw[i] for i in range(3)]}")
        print(f"    Raw supply-valid: {raw_ok}")
        print(f"    Repaired first allocation: {first_fixed}")
        print(f"    Repaired second allocation: {second_fixed}")
        print(f"    Repaired supply-valid: {fixed_ok}")

    total = len(rows)
    print("\nFINAL SUMMARY")
    print(f"Examples evaluated: {total}")
    print(f"Raw strict-format validity: {raw_format}/{total} "
          f"({100 * raw_format / total:.1f}%)")
    print(f"Raw supply-valid allocations: {raw_feasible}/{total} "
          f"({100 * raw_feasible / total:.1f}%)")
    print(f"Repair applied to: {repair_coverage}/{total} examples")
    print(f"Repaired supply-valid allocations: "
          f"{repaired_feasible}/{total} "
          f"({100 * repaired_feasible / total:.1f}%)")
    print(
        "\nInterpretation: repaired feasibility measures the post-processing "
        "rule, not the model's raw allocation accuracy or negotiation quality."
    )
    print("Diagnostic only; not a MACL continual-learning result.")


if __name__ == "__main__":
    main()
