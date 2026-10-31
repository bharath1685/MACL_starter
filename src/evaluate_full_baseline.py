
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sft" / "test.jsonl"
ADAPTER = ROOT / "outputs" / "sft_full_baseline"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LIMIT = 1052

ITEM_RE = re.compile(r"item([0-2])=(-?\d+)")


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.float16 if device == "mps" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, ADAPTER).to(device)
    model.eval()

    with DATA.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ][:LIMIT]

    valid_format = 0
    valid_counts = 0
    exact_match = 0

    for index, row in enumerate(rows, start=1):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"].strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        matches = ITEM_RE.findall(answer)
        target = ITEM_RE.findall(row["answer"])
        exact_match += int(matches == target and len(target) == 6)

        format_ok = (
            len(matches) == 6
            and [int(item[0]) for item in matches]
            == [0, 1, 2, 0, 1, 2]
        )
        valid_format += int(format_ok)

        count_match = re.search(
            r"Your item counts: \[(\d+), (\d+), (\d+)\]",
            row["prompt"],
        )
        partner_match = re.search(
            r"Other agent's item counts: \[(\d+), (\d+), (\d+)\]",
            row["prompt"],
        )

        counts_ok = False

        if format_ok and count_match and partner_match:
            own_counts = list(map(int, count_match.groups()))
            partner_counts = list(map(int, partner_match.groups()))
            allocation = [int(item[1]) for item in matches]

            counts_ok = all(
                allocation[i] >= 0
                and allocation[i + 3] >= 0
                and allocation[i] + allocation[i + 3]
                <= own_counts[i]
                for i in range(3)
            )

        valid_counts += int(counts_ok)

        print(f"\nExample {index}")
        print("Generated:", answer.strip()[:250])
        print("Format valid:", format_ok)
        print("Counts valid:", counts_ok)

    print("\n--- Summary ---")
    print("Examples:", len(rows))
    print("Valid allocation format:", f"{valid_format}/{len(rows)}")
    print("Stock-feasible allocations:", f"{valid_counts}/{len(rows)}")
    print("Exact target matches:", f"{exact_match}/{len(rows)}")


if __name__ == "__main__":
    main()
