#!/usr/bin/env python3
"""
Dataset outcome classification and reward-invariant audit.

Read-only: does not modify the dataset, reward engine, or model checkpoints.
Separates non-deal/rejection outcomes from malformed allocation outputs.
"""
import collections
import json
import re
from pathlib import Path

from reward_engine import (
    parse_input, validate_allocation, payoff,
    self_interested_rewards, fairness_shaped_rewards,
)

TAG = re.compile(r"<(input|partner_input|output)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
ASSIGNMENT = re.compile(r"item\s*\d+\s*=\s*(\d+)", re.IGNORECASE)


def tags_of(line):
    return {m.group(1).lower(): m.group(2).strip() for m in TAG.finditer(line)}


def classify_output(output):
    text = (output or "").strip().lower()
    if not text:
        return "empty_output"
    if "disconnect" in text:
        return "disconnect_termination"
    if any(token in text for token in ("disagree", "no deal", "no_deal", "no_agreement")):
        return "nondeal_or_rejection"
    vals = ASSIGNMENT.findall(output or "")
    if len(vals) == 6:
        return "allocation_candidate"
    return "other_unrecognized_output"


def parse_alloc(output):
    vals = [int(v) for v in ASSIGNMENT.findall(output or "")]
    if len(vals) != 6:
        return None
    return vals[:3], vals[3:]


def audit_file(path, gap_weight):
    c = collections.Counter()
    examples = collections.defaultdict(list)
    payoffs, gaps, fair_rewards = [], [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            c["records"] += 1
            tags = tags_of(line)
            if not all(k in tags for k in ("input", "partner_input", "output")):
                c["missing_required_tag"] += 1
                continue
            kind = classify_output(tags["output"])
            c[kind] += 1
            if kind != "allocation_candidate":
                if len(examples[kind]) < 3:
                    examples[kind].append({
                        "line": line_no,
                        "output": tags["output"][:180],
                    })
                continue
            try:
                scenario = parse_input(tags["input"], tags["partner_input"])
                allocation = parse_alloc(tags["output"])
                own_alloc, partner_alloc = allocation
                valid = validate_allocation(scenario, own_alloc, partner_alloc)
            except Exception as exc:
                c["allocation_parse_or_validation_exception"] += 1
                if len(examples["allocation_exception"]) < 3:
                    examples["allocation_exception"].append({
                        "line": line_no, "error": str(exc),
                        "output": tags["output"][:180],
                    })
                continue
            if not valid:
                c["invalid_allocation"] += 1
                continue
            c["valid_allocation"] += 1
            own = payoff(scenario.own_values, own_alloc)
            partner = payoff(scenario.partner_values, partner_alloc)
            _, _ = self_interested_rewards(own, partner)
            fair_own, fair_partner = fairness_shaped_rewards(
                own, partner, gap_weight=gap_weight
            )
            payoffs.append((float(own), float(partner)))
            gaps.append(float(abs(own - partner)))
            fair_rewards.extend([float(fair_own), float(fair_partner)])
    return {
        "file": str(path),
        "counts": dict(c),
        "mean_own_payoff": round(sum(x[0] for x in payoffs)/len(payoffs), 6) if payoffs else None,
        "mean_partner_payoff": round(sum(x[1] for x in payoffs)/len(payoffs), 6) if payoffs else None,
        "mean_absolute_payoff_gap": round(sum(gaps)/len(gaps), 6) if gaps else None,
        "mean_illustrative_fairness_reward_per_agent": round(sum(fair_rewards)/len(fair_rewards), 6) if fair_rewards else None,
        "examples": dict(examples),
    }


def test_reward_invariants(gap_weight):
    # Explicit synthetic cases test arithmetic invariants independently of dataset parsing.
    cases = [(0, 0), (5, 5), (8, 3), (3, 8), (10, 1)]
    results = []
    for own, partner in cases:
        r_own, r_partner = self_interested_rewards(own, partner)
        f_own, f_partner = fairness_shaped_rewards(own, partner, gap_weight=gap_weight)
        gap = abs(own - partner)
        assert r_own == own and r_partner == partner, "Self-interest reward invariant failed"
        assert abs(f_own - (own - gap_weight * gap)) < 1e-8
        assert abs(f_partner - (partner - gap_weight * gap)) < 1e-8
        results.append({
            "payoffs": [own, partner],
            "gap": gap,
            "self_interested_rewards": [r_own, r_partner],
            "illustrative_fairness_rewards": [f_own, f_partner],
        })
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_source/end-to-end-negotiator/src/data/negotiate")
    ap.add_argument("--gap-weight", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/task_objective_audit_v2.json")
    args = ap.parse_args()
    if args.gap_weight < 0:
        raise ValueError("gap weight must be nonnegative")
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    root = Path(args.data_dir)
    report = {
        "read_only": True,
        "gap_weight": args.gap_weight,
        "gap_weight_status": "explicit project setting; not numerically specified by the paper",
        "reward_invariant_tests": test_reward_invariants(args.gap_weight),
        "files": [],
    }
    for split in ("train.txt", "val.txt", "test.txt"):
        path = root / split
        if path.exists():
            report["files"].append(audit_file(path, args.gap_weight))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DATASET OUTCOME CLASSIFICATION + REWARD INVARIANTS")
    print("Non-deal/rejection outputs are counted separately from malformed allocation outputs.")
    print("Fairness gap weight is a project setting, not a paper-specified numeric value.")
    for item in report["files"]:
        print(f"\n{item['file']}")
        print("Counts:", item["counts"])
        print("Mean own payoff:", item["mean_own_payoff"])
        print("Mean partner payoff:", item["mean_partner_payoff"])
        print("Mean payoff gap:", item["mean_absolute_payoff_gap"])
        print("Examples of non-allocation outputs:", item["examples"].get("nondeal_or_rejection", []))
    print("\nReward invariant tests: PASS")
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
