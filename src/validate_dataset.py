from pathlib import Path
from collections import Counter
import re
import json

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data_source"
    / "end-to-end-negotiator"
    / "src"
    / "data"
    / "negotiate"
)

FILES = ["train.txt", "val.txt", "test.txt"]
TAG_PATTERN = re.compile(r"<([a-z_]+)>(.*?)</\1>", re.DOTALL)

def inspect_record(line):
    fields = {
        tag: value.strip()
        for tag, value in TAG_PATTERN.findall(line)
    }

    result = {
        "has_input": "input" in fields,
        "has_dialogue": "dialogue" in fields,
        "has_output": "output" in fields,
        "has_partner_input": "partner_input" in fields,
        "input_numbers": len(fields.get("input", "").split()),
        "dialogue_chars": len(fields.get("dialogue", "")),
        "output": fields.get("output", ""),
    }

    # The expected six-number input format is checked, not assumed valid.
    result["input_has_six_numbers"] = (
        result["input_numbers"] == 6
        and all(
            re.fullmatch(r"-?\d+", x)
            for x in fields.get("input", "").split()
        )
    )

    result["output_has_disagree"] = (
        "<disagree>" in result["output"]
    )
    result["output_has_selection"] = (
        "<selection>" in fields.get("dialogue", "")
    )

    return result, fields

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    print("Dataset:", DATA_DIR)

    for filename in FILES:
        path = DATA_DIR / filename
        records = []
        examples = []

        with path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                result, fields = inspect_record(line)
                records.append(result)

                if len(examples) < 2:
                    examples.append({
                        "line": line_number,
                        "input": fields.get("input", ""),
                        "output": fields.get("output", "")[:250],
                        "dialogue_preview": fields.get("dialogue", "")[:180],
                    })

        print(f"\n--- {filename} ---")
        print("Records inspected:", len(records))

        checks = [
            "has_input",
            "has_dialogue",
            "has_output",
            "has_partner_input",
            "input_has_six_numbers",
        ]

        for check in checks:
            passed = sum(r[check] for r in records)
            print(f"{check}: {passed}/{len(records)}")

        print("Output types:", dict(Counter(
            "disagree" if r["output_has_disagree"]
            else "other"
            for r in records
        )))

        print("Examples:")
        print(json.dumps(examples, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
