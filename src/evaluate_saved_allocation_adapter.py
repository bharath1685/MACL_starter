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

    if not (OUTPUT_DIR / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Saved adapter not found: {OUTPUT_DIR}")

    rows = read_valid_allocations(VAL_FILE)[:32]
    if not rows:
        raise RuntimeError("No valid validation allocations found.")

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    model.to("mps")
    model.eval()

    format_ok = 0
    counts_ok = 0

    print(f"\nEvaluating saved adapter on {len(rows)} validation examples.")
    print("No training will be performed.\n")

    for i, row in enumerate(rows, start=1):
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

        match = EXACT_FORMAT.fullmatch(generated)
        own_match = COUNTS_RE.search(row["prompt"])
        if not own_match:
            raise ValueError(f"Could not read item counts in example {i}.")

        available = list(map(int, own_match.groups()))
        format_valid = match is not None
        format_ok += int(format_valid)

        counts_valid = False
        if match:
            values = list(map(int, match.groups()))
            counts_valid = all(
                values[j] + values[j + 3] == available[j]
                for j in range(3)
            )
            counts_ok += int(counts_valid)

        print(
            f"{i:02d}. format={format_valid}, "
            f"counts={counts_valid}, available={available}"
        )
        print(f"    Generated: {generated!r}")

    print("\nEVALUATION SUMMARY")
    print(f"Examples: {len(rows)}")
    print(f"Strict format valid: {format_ok}/{len(rows)}")
    print(f"Count-consistent: {counts_ok}/{len(rows)}")
    print("Diagnostic only; not a MACL experimental result.")

if __name__ == "__main__":
    main()
