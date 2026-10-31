#!/usr/bin/env python3
"""Summarize matched MACL runs without treating pilot results as proof."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 41)
METHODS = ("naive", "ewc", "macl")


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def load_run(method, seed):
    path = (
        ROOT
        / "outputs"
        / f"macl_sft_{method}_v5_matched_s{seed}"
        / "metrics.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed run: {path}")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("method") != method or metrics.get("seed") != seed:
        raise ValueError(f"Run metadata does not match expected method/seed: {path}")
    return metrics


def average_agent_loss(stage, key):
    values = [agent[key] for agent in stage.values()]
    if not values or any(not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"Missing numeric validation loss for {key}")
    return statistics.mean(values)


def analyze_llm_runs():
    runs = {
        (method, seed): load_run(method, seed)
        for seed in SEEDS
        for method in METHODS
    }
    per_seed = []
    method_metrics = {method: [] for method in METHODS}

    for seed in SEEDS:
        seed_runs = {method: runs[(method, seed)] for method in METHODS}
        configs = {
            (
                metrics["steps_per_task"],
                metrics["train_examples"],
                metrics["fisher_examples"],
            )
            for metrics in seed_runs.values()
        }
        if len(configs) != 1:
            raise ValueError(f"Methods do not have matched budgets for seed {seed}")

        before_a = {
            method: seed_runs[method]["stages"]["before_A"]
            for method in METHODS
        }
        reference = before_a["naive"]
        for method in ("ewc", "macl"):
            for agent in ("agent0", "agent1"):
                for task in ("task_a_val_loss", "task_b_val_loss"):
                    if abs(
                        before_a[method][agent][task] - reference[agent][task]
                    ) > 1e-8:
                        raise ValueError(
                            f"Initial models differ across methods for seed {seed}"
                        )

        row = {"seed": seed, "methods": {}}
        for method in METHODS:
            stages = seed_runs[method]["stages"]
            task_a_after_a = average_agent_loss(
                stages["after_A"], "task_a_val_loss"
            )
            task_a_after_b = average_agent_loss(
                stages["after_B"], "task_a_val_loss"
            )
            task_b_after_b = average_agent_loss(
                stages["after_B"], "task_b_val_loss"
            )
            result = {
                "task_a_loss_after_A": task_a_after_a,
                "task_a_loss_after_B": task_a_after_b,
                "task_a_loss_change_after_B": task_a_after_b - task_a_after_a,
                "task_b_loss_after_B": task_b_after_b,
            }
            row["methods"][method] = result
            method_metrics[method].append(result)

        row["macl_minus_ewc_task_a_change"] = (
            row["methods"]["macl"]["task_a_loss_change_after_B"]
            - row["methods"]["ewc"]["task_a_loss_change_after_B"]
        )
        per_seed.append(row)

    summary = {}
    for method, records in method_metrics.items():
        summary[method] = {
            key: mean_std([record[key] for record in records])
            for key in records[0]
        }
    paired = [row["macl_minus_ewc_task_a_change"] for row in per_seed]
    return {
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "runs": summary,
        "paired_macl_minus_ewc_retention_change": mean_std(paired),
        "macl_better_seed_count": sum(value < 0 for value in paired),
        "per_seed": per_seed,
        "metric_definition": (
            "Task-A retention change is Task-A validation NLL after Task B "
            "minus Task-A validation NLL after Task A; lower is better."
        ),
    }


def analyze_toy_runs():
    path = ROOT / "results" / "toy_v2_reward_matrix.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        key = (int(row["seed"]), row["method"], int(row["stage"]))
        values[key] = float(row["score_A"])

    seeds = sorted({seed for seed, _, _ in values})
    methods = ("naive", "fixed_ewc", "macl_toy")
    forgetting = {
        method: [
            values[(seed, method, 2)] - values[(seed, method, 1)]
            for seed in seeds
        ]
        for method in methods
    }
    summary = {
        method: mean_std(scores)
        for method, scores in forgetting.items()
    }
    paired_macl_ewc = [
        macl - ewc
        for macl, ewc in zip(forgetting["macl_toy"], forgetting["fixed_ewc"])
    ]
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "task_a_score_change_after_B": summary,
        "paired_macl_minus_ewc_score_change": mean_std(paired_macl_ewc),
        "macl_better_seed_count": sum(value > 0 for value in paired_macl_ewc),
        "metric_definition": (
            "Change in Task-A negative MSE after Task B; higher (less negative) "
            "is better. This is a synthetic regression diagnostic."
        ),
    }


def audit_task_data_overlap():
    data_dir = ROOT / "data" / "continual_tasks"

    def prompts(filename):
        path = data_dir / filename
        with path.open(encoding="utf-8") as handle:
            return [
                json.loads(line)["prompt"]
                for line in handle
                if line.strip()
            ]

    a_train = prompts("task_a_train.jsonl")
    a_val = prompts("task_a_val.jsonl")
    b_train = prompts("task_b_train.jsonl")
    b_val = prompts("task_b_val.jsonl")
    a_all, b_all = set(a_train + a_val), set(b_train + b_val)
    return {
        "task_a_unique_prompts": len(a_all),
        "task_b_unique_prompts": len(b_all),
        "cross_task_prompt_overlap": len(a_all & b_all),
        "cross_task_overlap_fraction_of_task_a": (
            len(a_all & b_all) / len(a_all) if a_all else 0.0
        ),
    }


def write_report(llm, toy):
    out_json = ROOT / "results" / "macl_hypothesis_analysis.json"
    out_md = ROOT / "results" / "MACL_HYPOTHESIS_CONCLUSION.md"
    controlled_path = ROOT / "results" / "controlled_macl_study.json"
    scaling_path = ROOT / "results" / "multiagent_scaling_study.json"
    controlled = (
        json.loads(controlled_path.read_text(encoding="utf-8"))
        if controlled_path.is_file()
        else None
    )
    scaling = (
        json.loads(scaling_path.read_text(encoding="utf-8"))
        if scaling_path.is_file()
        else None
    )

    lines = [
        "# MACL and catastrophic forgetting: evidence-based conclusion",
        "",
        "## Conclusion",
        "",
        "The current evidence does **not** establish that multi-agent "
        "communication mitigates catastrophic forgetting better than fixed "
        "EWC. The controlled equal-search-budget synthetic study below found "
        "lower Task-A forgetting with MACL, but worse Task-B learning and "
        "worse overall balanced error than validation-tuned fixed EWC. "
        "The LLM results are a small, overlapping-task pilot, not a "
        "confirmation or a reproduction of the MACL paper.",
        "",
        "## Matched LLM pilot (confounded)",
        "",
        "The three seeds use matched initialization and training budgets, but "
        "the runs use the original Task A/B data, which have "
        f"{llm['data_audit']['cross_task_prompt_overlap']:,} exact prompts in "
        f"common ({llm['data_audit']['cross_task_overlap_fraction_of_task_a']:.1%} "
        "of Task A's unique prompts). Thus this is a confounded pilot, not "
        "evidence from independent tasks. Lower validation NLL is better. "
        "Retention change is Task-A NLL after Task B minus Task-A NLL after "
        "Task A; negative values mean Task A improved rather than being "
        "forgotten. Values are mean +/- sample standard deviation over seeds.",
        "",
        "| Method | Task-A NLL after A | Task-A NLL after B | Task-A change after B | Task-B NLL after B |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        data = llm["runs"][method]
        fields = (
            "task_a_loss_after_A",
            "task_a_loss_after_B",
            "task_a_loss_change_after_B",
            "task_b_loss_after_B",
        )
        cells = [
            f'{data[key]["mean"]:.5f} +/- {data[key]["std"]:.5f}'
            for key in fields
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    paired = llm["paired_macl_minus_ewc_retention_change"]
    lines.extend(
        [
            "",
            f"Paired MACL-minus-EWC Task-A change: "
            f"{paired['mean']:.5f} +/- {paired['std']:.5f}; MACL has a lower "
            f"(better) change on {llm['macl_better_seed_count']}/"
            f"{llm['seed_count']} seeds. This pilot is confounded by task "
            "overlap and only three seeds, so it cannot establish a reliable effect.",
            "",
            "## Five-seed synthetic diagnostic",
            "",
            "Task-A score change after learning Task B (negative MSE; higher "
            "is better):",
            "",
            "| Method | Mean change +/- sample SD |",
            "|---|---:|",
        ]
    )
    for method in ("naive", "fixed_ewc", "macl_toy"):
        data = toy["task_a_score_change_after_B"][method]
        lines.append(
            f"| {method} | {data['mean']:.5f} +/- {data['std']:.5f} |"
        )
    paired_toy = toy["paired_macl_minus_ewc_score_change"]
    lines.extend(
        [
            "",
            f"MACL-minus-fixed-EWC paired score change: "
            f"{paired_toy['mean']:.5f} +/- {paired_toy['std']:.5f}; "
            f"MACL is higher on {toy['macl_better_seed_count']}/"
            f"{toy['seed_count']} seeds. Both methods modestly outperform "
            "naive learning on this deliberately conflicting synthetic task, "
            "but substantial forgetting remains and their difference is "
            "negligible.",
            "",
            "The mean retention change is -17.84993 for naive, -17.17706 "
            "for fixed EWC, and -17.17398 for MACL. Thus consolidation "
            "improves the score by about 0.67 over naive, while the MACL "
            "increment over fixed EWC is only 0.00308 (about 0.02% of the "
            "EWC forgetting magnitude).",
            "",
            "## Limitations and interpretation",
            "",
            "- The LLM task labels are reward-ranked historical SFT proxies, "
            "not interactive negotiation tasks or online MARL.",
            "- A cleaned exact-prompt-disjoint dataset exists, but these "
            "completed LLM pilots did not use it; even that split would not "
            "guarantee distinct scenarios or genuinely conflicting objectives.",
            "- The MACL arm exchanges a scalar adapter-drift signal to set an "
            "EWC coefficient; this is a project prototype, not a verified "
            "implementation of the paper's algorithm.",
            "- Three LLM seeds and short runs are insufficient for a "
            "confirmatory claim, and the reported LLM pilot has substantial "
            "exact-prompt overlap.",
            "- Task A-prime performance is not used as a forgetting measure "
            "because Task A is trained again during that stage.",
            "",
            "## Verdict",
            "",
            "**Hypothesis status: not demonstrated by the current project "
            "evidence.** Current results support that EWC-style "
            "consolidation can reduce forgetting in a toy setting; they do "
            "not show a meaningful, reliable benefit from adding "
            "communication over a matched fixed-EWC baseline. A larger "
            "pre-registered, multi-seed study on genuinely distinct tasks, "
            "with tuned fixed-EWC controls and negotiation reward/retention "
            "metrics, is required before claiming that MACL works.",
            "",
            f"Machine-readable metrics: [macl_hypothesis_analysis.json]"
            f"(./macl_hypothesis_analysis.json).",
            "",
        ]
    )
    if controlled is not None:
        test = controlled["test_results"]
        differences = test["paired_differences"]
        balanced = differences["macl_minus_fixed_balanced_mse"]
        retention = differences["macl_minus_fixed_task_a_retention_change"]
        task_b = differences["macl_minus_fixed_task_b_mse"]
        comm_vs_self = differences["macl_minus_self_signal_balanced_mse"]
        section = [
            "",
            "## Controlled synthetic study: 100 held-out seeds",
            "",
            "This study tunes fixed EWC and the adaptive MACL mapping on "
            "32 separate validation seeds, with an equal 108-candidate "
            "hyperparameter search for each. It then evaluates on 100 "
            "unseen seeds. Lower errors are better.",
            "",
            "| Paired difference (MACL minus comparator) | Mean | Bootstrap 95% CI |",
            "|---|---:|---:|",
            f"| Balanced test MSE vs tuned fixed EWC | "
            f"{balanced['mean']:.5f} | "
            f"[{balanced['ci95_bootstrap'][0]:.5f}, "
            f"{balanced['ci95_bootstrap'][1]:.5f}] |",
            f"| Task-A forgetting vs tuned fixed EWC | "
            f"{retention['mean']:.5f} | "
            f"[{retention['ci95_bootstrap'][0]:.5f}, "
            f"{retention['ci95_bootstrap'][1]:.5f}] |",
            f"| Task-B test MSE vs tuned fixed EWC | "
            f"{task_b['mean']:.5f} | "
            f"[{task_b['ci95_bootstrap'][0]:.5f}, "
            f"{task_b['ci95_bootstrap'][1]:.5f}] |",
            f"| Balanced test MSE vs self-signal ablation | "
            f"{comm_vs_self['mean']:.5f} | "
            f"[{comm_vs_self['ci95_bootstrap'][0]:.5f}, "
            f"{comm_vs_self['ci95_bootstrap'][1]:.5f}] |",
            "",
            "Interpretation: MACL retained Task A better than tuned EWC "
            "on average, but Task-B error increased enough that its "
            "balanced test MSE was significantly worse. Partner "
            "communication did not outperform the self-signal ablation "
            "(its balanced-error interval includes zero). Therefore this "
            "experiment does **not** show MACL outperforms fixed EWC, and "
            "does not isolate a reliable benefit from multi-agent "
            "communication.",
            "",
            "Full protocol, ablations, and per-seed results: "
            "[CONTROLLED_MACL_STUDY.md](./CONTROLLED_MACL_STUDY.md) and "
            "[controlled_macl_study.json](./controlled_macl_study.json).",
            "",
        ]
        insertion = lines.index("## Limitations and interpretation")
        lines[insertion:insertion] = section
    if scaling is not None:
        rows = [
            "",
            "## Agent-count scaling: 100 held-out seeds per team size",
            "",
            "Every team size received 320 optimizer updates per task. EWC and "
            "adaptive methods each used 108 validation candidates over 24 "
            "separate seeds. Values below are paired peer-communication "
            "minus fixed-EWC balanced test MSE; positive values favor EWC.",
            "",
            "| Agents | Difference | Bootstrap 95% CI | Peer wins |",
            "|---:|---:|---:|---:|",
        ]
        peer_not_better = True
        for size in scaling["protocol"]["team_sizes"]:
            comparison = scaling["team_sizes"][str(size)]["test_results"][
                "paired_comparisons"
            ]["peer_minus_fixed_ewc_balanced_mse"]
            peer_not_better &= comparison["mean"] > 0
            rows.append(
                f"| {size} | {comparison['mean']:.5f} | "
                f"[{comparison['ci95_bootstrap'][0]:.5f}, "
                f"{comparison['ci95_bootstrap'][1]:.5f}] | "
                f"{comparison['peer_better_seed_count']}/"
                f"{comparison['seed_count']} |"
            )
        rows.extend(
            [
                "",
                "No tested multi-agent team size outperformed tuned fixed EWC "
                "on balanced test error; all paired intervals favor fixed "
                "EWC. Eight agents did outperform the local adaptive "
                "no-communication variant on balanced error, but not the "
                "shuffled-message ablation, and its Task-B error was higher. "
                "This does not show that communication itself caused the "
                "gain or that adding agents improves performance.",
                "",
                "Full scaling report: "
                "[MULTIAGENT_SCALING_STUDY.md](./MULTIAGENT_SCALING_STUDY.md) "
                "and [multiagent_scaling_study.json]"
                "(./multiagent_scaling_study.json).",
            ]
        )
        insertion = lines.index("## Limitations and interpretation")
        lines[insertion:insertion] = rows
    out_md.write_text("\n".join(lines), encoding="utf-8")
    combined = {
        "matched_overlapping_llm_pilot": llm,
        "synthetic_toy": toy,
    }
    if controlled is not None:
        combined["controlled_synthetic_study"] = controlled
    if scaling is not None:
        combined["multiagent_scaling_study"] = scaling
    out_json.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


def main():
    llm = analyze_llm_runs()
    llm["data_audit"] = audit_task_data_overlap()
    write_report(llm, analyze_toy_runs())


if __name__ == "__main__":
    main()
