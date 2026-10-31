#!/usr/bin/env python3
"""
Read-only audit of the Deal-or-No-Deal task objectives.

This script does NOT train models or modify datasets. It checks whether the
current reward_engine can compute self-interested and illustrative
fairness-shaped rewards from the original source records.

Important: MACL.pdf specifies the task sequence conceptually, but does not
specify a numerical fairness-gap coefficient. The current reward_engine's
gap_weight=1.0 is therefore treated as an explicit project choice for audit,
not as a formula recovered from the paper.
"""
import collections
import json
import re
from pathlib import Path

from reward_engine import (
    parse_input,
    validate_allocation,
    payoff,
    self_interested_rewards,
    fairness_shaped_rewards,
)

TAG = re.compile(r"<(input|partner_input|output)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
ASSIGNMENT = re.compile(r"item\s*\d+\s*=\s*(\d+)", re.IGNORECASE)


def extract_tags(line):
    found = {m.group(1).lower(): m.group(2).strip() for m in TAG.finditer(line)}
    return found


def parse_output(output_text):
    vals = [int(x) for x in ASSIGNMENT.findall(output_text or "")]
    if len(vals) != 6:
        return None
    return vals[:3], vals[3:]


def audit_file(path, limit=None, gap_weight=1.0):
    path = Path(path)
    if not path.exists():
        print(f"SKIP missing file: {path}")
        return None

    counts = collections.Counter()
    own_payoffs, partner_payoffs, gaps = [], [], []
    self_rewards, fair_own, fair_partner = [], [], []
    examples = []
    records = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records += 1
            if limit is not None and records > limit:
                break
            tags = extract_tags(line)
            if not all(k in tags for k in ("input", "partner_input", "output")):
                counts["missing_required_tag"] += 1
                continue
            try:
                scenario = parse_input(tags["input"], tags["partner_input"])
            except Exception:
                counts["input_parse_error"] += 1
                continue
            allocations = parse_output(tags["output"])
            if allocations is None:
                counts["output_parse_error"] += 1
                continue
            own_alloc, partner_alloc = allocations
            try:
                valid = validate_allocation(scenario, own_alloc, partner_alloc)
            except Exception as e:
                counts["allocation_validation_error"] += 1
                continue
            counts["parsed"] += 1
            if not valid:
                counts["invalid_allocation"] += 1
                continue
            counts["valid_allocation"] += 1
            own = payoff(scenario.own_values, own_alloc)
            partner = payoff(scenario.partner_values, partner_alloc)
            self_r, _ = self_interested_rewards(own, partner)
            fair_r_own, fair_r_partner = fairness_shaped_rewards(
                own, partner, gap_weight=gap_weight
            )
            own_payoffs.append(float(own))
            partner_payoffs.append(float(partner))
            gaps.append(float(abs(own - partner)))
            self_rewards.append(float(self_r))
            fair_own.append(float(fair_r_own))
            fair_partner.append(float(fair_r_partner))
            if len(examples) < 3:
                examples.append({
                    "own_payoff": own,
                    "partner_payoff": partner,
                    "absolute_payoff_gap": abs(own - partner),
                    "self_interested_reward_agent_1": self_r,
                    "illustrative_fairness_reward_agent_1": fair_r_own,
                    "illustrative_fairness_reward_agent_2": fair_r_partner,
                    "allocation_own": own_alloc,
                    "allocation_partner": partner_alloc,
                })

    def stats(xs):
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 6),
            "min": round(min(xs), 6),
            "max": round(max(xs), 6),
        }

    result = {
        "file": str(path),
        "records_scanned": records if limit is None else min(records, limit),
        "counts": dict(counts),
        "own_payoff": stats(own_payoffs),
        "partner_payoff": stats(partner_payoffs),
        "absolute_payoff_gap": stats(gaps),
        "self_interested_reward_agent_1": stats(self_rewards),
        "illustrative_fairness_reward_agent_1": stats(fair_own),
        "illustrative_fairness_reward_agent_2": stats(fair_partner),
        "gap_weight_used_for_audit_only": gap_weight,
        "examples": examples,
    }
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_source/end-to-end-negotiator/src/data/negotiate")
    ap.add_argument("--limit", type=int, default=0, help="0 scans the full file")
    ap.add_argument("--gap-weight", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/task_objective_audit.json")
    args = ap.parse_args()

    if args.gap_weight < 0:
        raise ValueError("--gap-weight must be nonnegative")
    root = Path(args.data_dir)
    limit = args.limit or None
    report = {
        "read_only": True,
        "paper_task_sequence": [
            "Task 1: self-interested own item-value payoff",
            "Task 2: fairness-shaped reward minimizing the payoff gap",
            "Task 3: self-interested reward again",
        ],
        "source_limitation": (
            "The paper describes the fairness objective conceptually but does not specify "
            "the numeric gap coefficient. The current fairness_shaped_rewards implementation "
            "is an illustrative operationalization, not a paper-verified equation."
        ),
        "files": [],
    }
    for name in ("train.txt", "val.txt", "test.txt"):
        item = audit_file(root / name, limit=limit, gap_weight=args.gap_weight)
        if item is not None:
            report["files"].append(item)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Do not overwrite an existing report.
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("READ-ONLY TASK OBJECTIVE AUDIT")
    print("Task 1 = own payoff; Task 2 = illustrative payoff-minus-gap; Task 3 = own payoff.")
    print("Fairness gap weight is a project audit setting, not specified numerically by the paper.")
    for item in report["files"]:
        print(f"\n{item['file']}")
        print("Counts:", item["counts"])
        print("Own payoff:", item["own_payoff"])
        print("Partner payoff:", item["partner_payoff"])
        print("Absolute payoff gap:", item["absolute_payoff_gap"])
        print("Illustrative fairness reward (agent 1):", item["illustrative_fairness_reward_agent_1"])
        print("Examples:", json.dumps(item["examples"], indent=2))
    print(f"\nSaved report: {out}")


if __name__ == "__main__":
    main()
