
from pathlib import Path
import re
from collections import Counter

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

def repeated_tag(output, tag):
    return bool(re.fullmatch(rf"(<{tag}>\s*)+", output))

def main():
    checked = 0
    outcomes = Counter()
    errors = []

    with DATA_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                own_input = tuple(
                    int(x) for x in extract(line, "input").split()
                )
                partner_input = tuple(
                    int(x) for x in extract(line, "partner_input").split()
                )
                output = extract(line, "output")

                if len(own_input) != 6 or len(partner_input) != 6:
                    raise ValueError("Expected six input integers")

                counts = own_input[0::2]

                if counts != partner_input[0::2]:
                    raise ValueError("Input item counts do not match")

                if any(x < 0 for x in own_input + partner_input):
                    raise ValueError("Negative input value")

                if repeated_tag(output, "disagree"):
                    outcomes["disagree"] += 1

                elif repeated_tag(output, "no_agreement"):
                    outcomes["no_agreement"] += 1

                elif repeated_tag(output, "disconnect"):
                    outcomes["disconnect"] += 1

                else:
                    pairs = re.findall(r"(item[0-2])=(\d+)", output)
                    labels = [label for label, value in pairs]
                    expected = [
                        "item0", "item1", "item2",
                        "item0", "item1", "item2"
                    ]

                    if labels != expected:
                        raise ValueError("Unrecognized output format")

                    allocations = [int(value) for label, value in pairs]
                    own_allocation = allocations[:3]
                    partner_allocation = allocations[3:]

                    if any(
                        own_allocation[i] + partner_allocation[i] != counts[i]
                        for i in range(3)
                    ):
                        raise ValueError(
                            "Allocations do not match available item counts"
                        )

                    outcomes["valid deal allocation"] += 1

                checked += 1

            except Exception as error:
                errors.append((line_number, str(error)))

    print(f"Records checked: {checked + len(errors)}")
    print(f"Records classified: {checked}")
    print(f"Records needing investigation: {len(errors)}")

    for outcome, count in sorted(outcomes.items()):
        print(f"{outcome}: {count}")

    for line_number, error in errors[:10]:
        print(f"Line {line_number}: {error}")

if __name__ == "__main__":
    main()
