#!/usr/bin/env python3
"""Build reward-ranked historical SFT tasks for a controlled CL pilot.

Task A: top-quartile recorded deals by own payoff.
Task B: top-quartile recorded deals by own payoff minus payoff gap.

This is a supervised proxy, not online RL or a verified reproduction of MACL.
Existing data and checkpoints are never modified.
"""
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data_source/end-to-end-negotiator/src/data/negotiate"
SFT = ROOT / "data/sft"
OUT = ROOT / "data/continual_tasks"
GAP_WEIGHT = 1.0
TOP_FRACTION = 0.25

def extract(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text)
    if not m:
        raise ValueError(f"Missing <{tag}>")
    return m.group(1).strip()

def parse_deal(output, counts):
    pairs = re.findall(r"(item[0-2])=(\d+)", output)
    if [x[0] for x in pairs] != [
        "item0", "item1", "item2", "item0", "item1", "item2"
    ]:
        return None
    a = [int(x[1]) for x in pairs[:3]]
    b = [int(x[1]) for x in pairs[3:]]
    if any(a[i] + b[i] != counts[i] for i in range(3)):
        return None
    return a, b

def load_split(split):
    raw_path = SOURCE / f"{split}.txt"
    sft_path = SFT / f"{split}.jsonl"
    with raw_path.open(encoding="utf-8") as f:
        raw = [x for x in f if x.strip()]
    with sft_path.open(encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    if len(raw) != len(rows):
        raise RuntimeError(
            f"{split}: source/SFT row count mismatch: {len(raw)} vs {len(rows)}"
        )

    records = []
    for i, (line, row) in enumerate(zip(raw, rows), 1):
        own = [int(x) for x in extract(line, "input").split()]
        partner = [int(x) for x in extract(line, "partner_input").split()]
        output = extract(line, "output")
        if len(own) != 6 or len(partner) != 6:
            raise ValueError(f"{split} row {i}: expected six input integers")

        counts = own[0::2]
        own_values = own[1::2]
        partner_counts = partner[0::2]
        partner_values = partner[1::2]
        if counts != partner_counts:
            raise ValueError(f"{split} row {i}: item counts differ")

        allocation = parse_deal(output, counts)
        if allocation is None:
            continue

        # Verify the paired SFT target really matches this source allocation.
        target_nums = re.findall(r"item[0-2]=(\d+)", row["answer"])
        source_nums = re.findall(r"item[0-2]=(\d+)", output)
        if target_nums != source_nums or len(target_nums) != 6:
            raise RuntimeError(
                f"{split} row {i}: source/target alignment failed; aborting"
            )

        a, b = allocation
        own_pay = sum(v * q for v, q in zip(own_values, a))
        partner_pay = sum(v * q for v, q in zip(partner_values, b))
        gap = abs(own_pay - partner_pay)
        records.append({
            "row": row,
            "own_payoff": own_pay,
            "partner_payoff": partner_pay,
            "gap": gap,
            "task_a_score": float(own_pay),
            "task_b_score": float(own_pay - GAP_WEIGHT * gap),
        })
    return records, len(raw)

def select_top(records, score_key):
    if not records:
        raise RuntimeError(f"No valid deal records for {score_key}")
    ranked = sorted(
        records,
        key=lambda r: (r[score_key], r["own_payoff"], -r["gap"]),
        reverse=True,
    )
    n = max(1, int(len(ranked) * TOP_FRACTION))
    return [r["row"] for r in ranked[:n]], ranked

def main():
    if OUT.exists():
        raise SystemExit(
            f"{OUT} already exists. Refusing to overwrite any task data."
        )

    train, raw_train_n = load_split("train")
    val, raw_val_n = load_split("val")
    if not train or not val:
        raise RuntimeError("Training or validation has no valid deals.")

    train_a, ranked_a = select_top(train, "task_a_score")
    train_b, ranked_b = select_top(train, "task_b_score")
    val_a, ranked_va = select_top(val, "task_a_score")
    val_b, ranked_vb = select_top(val, "task_b_score")

    OUT.mkdir(parents=True)
    datasets = {
        "task_a_train.jsonl": train_a,
        "task_b_train.jsonl": train_b,
        "task_a_val.jsonl": val_a,
        "task_b_val.jsonl": val_b,
    }
    for name, rows in datasets.items():
        with (OUT / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "method": "reward-ranked historical supervised fine-tuning proxy",
        "not_online_rl": True,
        "not_verified_macl_reproduction": True,
        "task_a": "top quartile by own payoff",
        "task_b": "top quartile by own payoff - gap_weight * absolute payoff gap",
        "gap_weight": GAP_WEIGHT,
        "top_fraction": TOP_FRACTION,
        "raw_rows": {"train": raw_train_n, "validation": raw_val_n},
        "valid_deal_rows": {"train": len(train), "validation": len(val)},
        "selected_rows": {k: len(v) for k, v in datasets.items()},
        "mean_selected_own_payoff": {
            "task_a_train": statistics.mean(r["own_payoff"] for r in ranked_a[:len(train_a)]),
            "task_b_train": statistics.mean(r["own_payoff"] for r in ranked_b[:len(train_b)]),
        },
        "limitations": [
            "Task labels are reward-ranked proxies, not original paper labels.",
            "Partner private values are used offline to score recorded deals.",
            "Filtering recorded allocations does not create interactive negotiation.",
            "Hyperparameters require validation-only selection.",
        ],
    }
    with (OUT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Created:", OUT)
    print("Valid deals train/validation:",
          len(train), "/", len(val))
    for name, rows in datasets.items():
        print(f"{name}: {len(rows)}")
    print("Manifest:", OUT / "manifest.json")
    print("Existing checkpoints and datasets were not modified.")

if __name__ == "__main__":
    main()
