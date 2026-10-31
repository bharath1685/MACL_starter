"""Read-only audit of negotiation records for continual-learning tasks.

This script does not train a model or modify the source dataset.
Fairness keywords are only a diagnostic, not authoritative task labels.
"""
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_source/end-to-end-negotiator/src/data/negotiate"
sys.path.insert(0, str(ROOT / "src"))

from reward_engine import parse_input, payoff, validate_allocation


def extract(line, tag):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", line)
    if not match:
        raise ValueError(f"Missing <{tag}> tag")
    return match.group(1).strip()


def parse_output(text):
    if re.fullmatch(r"(<disagree>\s*)+", text):
        return None
    if re.fullmatch(r"(<no_agreement>\s*)+", text):
        return None
    if re.fullmatch(r"(<disconnect>\s*)+", text):
        return None

    pairs = re.findall(r"(item[0-2])=(\d+)", text)
    expected = [f"item{i}" for i in range(3)] * 2
    if [k for k, _ in pairs] != expected:
        raise ValueError("Unrecognized allocation output")

    values = [int(v) for _, v in pairs]
    return values[:3], values[3:]


def main():
    totals = Counter()
    fairness_words = ("fairness", "fair", "equal", "equitable")
    task_candidates = {"keyword_match": [], "no_keyword_match": []}
    errors = []

    for split in ("train", "val", "test"):
        path = DATA / f"{split}.txt"
        if not path.exists():
            errors.append((str(path), "file missing"))
            continue

        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                totals[f"{split}_records"] += 1

                try:
                    own_text = extract(line, "input")
                    partner_text = extract(line, "partner_input")
                    dialogue = extract(line, "dialogue")
                    output_text = extract(line, "output")

                    scenario = parse_input(own_text, partner_text)
                    allocation = parse_output(output_text)

                    if allocation is None:
                        totals[f"{split}_nondeal"] += 1
                        continue

                    own_alloc, partner_alloc = allocation
                    if not validate_allocation(
                        scenario, own_alloc, partner_alloc
                    ):
                        raise ValueError("Allocation violates item counts")

                    own_pay = payoff(scenario.own_values, own_alloc)
                    partner_pay = payoff(
                        scenario.partner_values, partner_alloc
                    )

                    totals[f"{split}_valid_deals"] += 1
                    totals[f"{split}_own_payoff_sum"] += own_pay
                    totals[f"{split}_partner_payoff_sum"] += partner_pay
                    totals[f"{split}_payoff_gap_sum"] += abs(
                        own_pay - partner_pay
                    )

                    record = {
                        "own_payoff": own_pay,
                        "partner_payoff": partner_pay,
                        "gap": abs(own_pay - partner_pay),
                    }
                    has_keyword = any(
                        word in dialogue.lower() for word in fairness_words
                    )
                    key = "keyword_match" if has_keyword else "no_keyword_match"
                    task_candidates[key].append(record)

                except Exception as exc:
                    totals[f"{split}_invalid"] += 1
                    if len(errors) < 20:
                        errors.append((f"{split}:{line_no}", str(exc)))

    print("NEGOTIATION DATA AUDIT")
    print("=====================")
    for key, value in sorted(totals.items()):
        print(f"{key}: {value}")

    for key, records in task_candidates.items():
        print(f"\nCandidate group: {key}")
        print(f"Valid deals: {len(records)}")
        if records:
            n = len(records)
            print(f"Mean own payoff: {sum(r['own_payoff'] for r in records)/n:.3f}")
            print(f"Mean partner payoff: {sum(r['partner_payoff'] for r in records)/n:.3f}")
            print(f"Mean payoff gap: {sum(r['gap'] for r in records)/n:.3f}")

    if errors:
        print("\nFirst errors:")
        for location, message in errors:
            print(f"{location}: {message}")

    print("\nIMPORTANT:")
    print("Keyword groups are not validated self-interest/fairness labels.")
    print("This audit does not train a model or establish MACL effectiveness.")


if __name__ == "__main__":
    main()
