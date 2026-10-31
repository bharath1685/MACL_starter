
from pathlib import Path
import re

DATA_FILE = Path(
    "data_source/end-to-end-negotiator/src/data/negotiate/train.txt"
)

def extract(line, tag):
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        line
    )
    if not match:
        raise ValueError(f"Missing <{tag}> field")
    return match.group(1).strip()

def parse_numbers(text):
    return tuple(int(x) for x in text.split())

def main():
    checked = 0
    errors = []

    with DATA_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                own = parse_numbers(extract(line, "input"))
                partner = parse_numbers(
                    extract(line, "partner_input")
                )

                if len(own) != 6 or len(partner) != 6:
                    raise ValueError("Expected six integers per input")

                own_counts = own[0::2]
                partner_counts = partner[0::2]

                if own_counts != partner_counts:
                    raise ValueError("Item counts do not match")

                if any(x < 0 for x in own + partner):
                    raise ValueError("Negative count or value")

                checked += 1

            except Exception as error:
                errors.append((line_number, str(error)))

    print(f"Records checked: {checked}")
    print(f"Invalid records: {len(errors)}")

    for line_number, error in errors[:10]:
        print(f"Line {line_number}: {error}")

if __name__ == "__main__":
    main()
