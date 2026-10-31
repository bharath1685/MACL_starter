
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "data_source"
    / "end-to-end-negotiator"
    / "src"
    / "data"
    / "negotiate"
)
OUTPUT_DIR = ROOT / "data" / "sft"
TAG_PATTERN = re.compile(r"<(input|dialogue|output|partner_input)>(.*?)</\1>")


def parse_record(line):
    fields = dict(TAG_PATTERN.findall(line))

    required = {"input", "dialogue", "output", "partner_input"}
    if not required.issubset(fields):
        raise ValueError("Record is missing a required field")

    own_tokens = [int(x) for x in fields["input"].split()]
    partner_tokens = [int(x) for x in fields["partner_input"].split()]

    if len(own_tokens) != 6 or len(partner_tokens) != 6:
        raise ValueError("Expected six numbers in each input")

    own_counts = own_tokens[0::2]
    own_values = own_tokens[1::2]
    partner_counts = partner_tokens[0::2]

    # Remove the end-of-negotiation selection marker from the prompt.
    dialogue = fields["dialogue"].replace("<selection>", "").strip()

    prompt = (
        "You are an agent negotiating over books, hats, and balls.\n"
        "Your private item counts and values are given below. "
        "The other agent's values are private to them.\n"
        f"Your item counts: {own_counts}\n"
        f"Your private values: {own_values}\n"
        f"Other agent's item counts: {partner_counts}\n"
        "Dialogue so far:\n"
        f"{dialogue}\n"
        "Predict the recorded outcome and allocation. "
        "Return the original output format exactly."
    )

    return {
        "prompt": prompt,
        "answer": fields["output"].strip(),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        source = DATA_DIR / f"{split}.txt"
        destination = OUTPUT_DIR / f"{split}.jsonl"

        total = 0
        with source.open(encoding="utf-8") as src, \
                destination.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                if not line.strip():
                    continue
                try:
                    record = parse_record(line)
                except Exception as exc:
                    raise ValueError(
                        f"{source.name}, line {line_number}: {exc}"
                    ) from exc

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

        print(f"{split}: wrote {total} records to {destination}")

    print("\nData preparation complete. Original files were not changed.")

if __name__ == "__main__":
    main()
