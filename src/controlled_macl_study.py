#!/usr/bin/env python3
"""Validation-tuned synthetic stress test for communicated EWC strength.

This is a controlled linear-regression diagnostic, not an LLM or MACL
paper reproduction. Hyperparameters are selected on validation seeds and
reported on separate test seeds to avoid tuning on the reported outcomes.
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
VALIDATION_SEEDS = tuple(range(1000, 1032))
TEST_SEEDS = tuple(range(2000, 2100))
AGENTS = 2
DIMENSIONS = 8
TRAIN_EXAMPLES = 128
TEST_EXAMPLES = 512
STEPS = 160
LEARNING_RATE = 0.035
NOISE = 0.08
FIXED_LAMBDAS = tuple(
    [0.0, *np.geomspace(0.01, 12.0, 107).tolist()]
)
ADAPTIVE_LOW = (0.0, 0.1, 0.3)
ADAPTIVE_HIGH = (0.3, 0.7, 1.5, 3.0, 6.0, 12.0)
ADAPTIVE_SCALE = (0.01, 0.1, 0.5, 1.0, 3.0, 10.0)


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def make_group(seed):
    """Make a paired group with shared task A and correlated task conflicts."""
    rng = np.random.default_rng(seed)
    task_a = _unit(rng.normal(size=DIMENSIONS)) * math.sqrt(DIMENSIONS)
    direction = rng.normal(size=DIMENSIONS)
    direction -= np.dot(direction, task_a) / np.dot(task_a, task_a) * task_a
    direction = _unit(direction) * math.sqrt(DIMENSIONS)

    shared_severity = rng.uniform(0.35, 1.65)
    agents = []
    for agent in range(AGENTS):
        severity = float(np.clip(shared_severity + rng.normal(0.0, 0.18), 0.1, 1.9))
        angle = severity * math.pi / 2
        task_b = math.cos(angle) * task_a + math.sin(angle) * direction
        data_rng = np.random.default_rng(seed * 31 + agent * 997 + 17)
        x_a = data_rng.normal(size=(TRAIN_EXAMPLES, DIMENSIONS))
        y_a = x_a @ task_a + data_rng.normal(scale=NOISE, size=TRAIN_EXAMPLES)
        x_b = data_rng.normal(size=(TRAIN_EXAMPLES, DIMENSIONS))
        y_b = x_b @ task_b + data_rng.normal(scale=NOISE, size=TRAIN_EXAMPLES)
        x_a_test = data_rng.normal(size=(TEST_EXAMPLES, DIMENSIONS))
        y_a_test = x_a_test @ task_a + data_rng.normal(
            scale=NOISE, size=TEST_EXAMPLES
        )
        x_b_test = data_rng.normal(size=(TEST_EXAMPLES, DIMENSIONS))
        y_b_test = x_b_test @ task_b + data_rng.normal(
            scale=NOISE, size=TEST_EXAMPLES
        )
        agents.append({
            "severity": severity,
            "train_a": (x_a, y_a),
            "train_b": (x_b, y_b),
            "test_a": (x_a_test, y_a_test),
            "test_b": (x_b_test, y_b_test),
        })
    return agents


def _mse(theta, data):
    x, y = data
    residual = x @ theta - y
    return float(np.mean(residual * residual))


def _fit(theta, data, fisher=None, anchor=None, strength=0.0):
    x, y = data
    theta = theta.copy()
    for _ in range(STEPS):
        residual = x @ theta - y
        grad = (2.0 / len(x)) * (x.T @ residual)
        if strength:
            grad += 2.0 * strength * fisher * (theta - anchor)
        theta -= LEARNING_RATE * grad
    return theta


def _prepare_group(seed):
    prepared = []
    for agent in make_group(seed):
        theta_a = _fit(np.zeros(DIMENSIONS), agent["train_a"])
        x_a, y_a = agent["train_a"]
        residual = x_a @ theta_a - y_a
        per_example_grad = 2.0 * x_a * residual[:, None]
        fisher = np.mean(per_example_grad**2, axis=0)
        mean = float(np.mean(fisher))
        if mean > 0:
            fisher = fisher / mean
        fisher = np.maximum(fisher, 1e-8)
        proposed_b = _fit(theta_a, agent["train_b"])
        signal = float(np.mean(fisher * (proposed_b - theta_a) ** 2))
        prepared.append({
            **agent,
            "theta_a": theta_a,
            "fisher": fisher,
            "signal": signal,
            "a_after_a": _mse(theta_a, agent["test_a"]),
        })
    return prepared


def _score(group, strengths):
    losses = []
    retention_changes = []
    for agent, strength in zip(group, strengths):
        theta_b = _fit(
            agent["theta_a"],
            agent["train_b"],
            agent["fisher"],
            agent["theta_a"],
            strength,
        )
        a_loss = _mse(theta_b, agent["test_a"])
        b_loss = _mse(theta_b, agent["test_b"])
        losses.append((a_loss, b_loss))
        retention_changes.append(a_loss - agent["a_after_a"])
    return {
        "balanced_test_mse": float(np.mean(losses)),
        "task_a_retention_change": float(np.mean(retention_changes)),
        "task_a_test_mse_after_b": float(np.mean([row[0] for row in losses])),
        "task_b_test_mse_after_b": float(np.mean([row[1] for row in losses])),
    }


def _evaluate_fixed(groups, strength):
    rows = [_score(group, (strength,) * AGENTS) for group in groups]
    return float(np.mean([row["balanced_test_mse"] for row in rows]))


def _adaptive_strength(signal, low, high, scale):
    fraction = max(0.0, signal) / (max(0.0, signal) + scale)
    return low + (high - low) * fraction


def _evaluate_adaptive(groups, low, high, scale, signal_mode="partner"):
    rows = []
    for group_index, group in enumerate(groups):
        signals = []
        for agent_index in range(AGENTS):
            if signal_mode == "partner":
                signal = group[1 - agent_index]["signal"]
            elif signal_mode == "self":
                signal = group[agent_index]["signal"]
            elif signal_mode == "shuffled":
                donor = groups[(group_index + 1) % len(groups)]
                signal = donor[1 - agent_index]["signal"]
            else:
                raise ValueError(f"Unknown signal mode: {signal_mode}")
            signals.append(_adaptive_strength(signal, low, high, scale))
        rows.append(_score(group, signals))
    return rows


def _bootstrap_mean_ci(values, seed=90210, samples=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _select_hyperparameters(validation_groups):
    fixed_scores = {
        strength: _evaluate_fixed(validation_groups, strength)
        for strength in FIXED_LAMBDAS
    }
    fixed = min(fixed_scores, key=fixed_scores.get)

    adaptive_scores = {}
    for low in ADAPTIVE_LOW:
        for high in ADAPTIVE_HIGH:
            if high < low:
                continue
            for scale in ADAPTIVE_SCALE:
                rows = _evaluate_adaptive(
                    validation_groups, low, high, scale, signal_mode="partner"
                )
                adaptive_scores[(low, high, scale)] = float(np.mean([
                    row["balanced_test_mse"] for row in rows
                ]))
    adaptive = min(adaptive_scores, key=adaptive_scores.get)
    return {
        "fixed_ewc_lambda": fixed,
        "fixed_validation_balanced_mse": fixed_scores[fixed],
        "fixed_validation_grid": {
            str(key): value for key, value in fixed_scores.items()
        },
        "macl_low_lambda": adaptive[0],
        "macl_high_lambda": adaptive[1],
        "macl_signal_scale": adaptive[2],
        "macl_validation_balanced_mse": adaptive_scores[adaptive],
        "adaptive_validation_candidate_count": len(adaptive_scores),
    }


def _summarize_test(groups, hyperparameters):
    fixed_rows = [
        _score(group, (hyperparameters["fixed_ewc_lambda"],) * AGENTS)
        for group in groups
    ]
    macl_rows = _evaluate_adaptive(
        groups,
        hyperparameters["macl_low_lambda"],
        hyperparameters["macl_high_lambda"],
        hyperparameters["macl_signal_scale"],
        signal_mode="partner",
    )
    self_rows = _evaluate_adaptive(
        groups,
        hyperparameters["macl_low_lambda"],
        hyperparameters["macl_high_lambda"],
        hyperparameters["macl_signal_scale"],
        signal_mode="self",
    )
    shuffled_rows = _evaluate_adaptive(
        groups,
        hyperparameters["macl_low_lambda"],
        hyperparameters["macl_high_lambda"],
        hyperparameters["macl_signal_scale"],
        signal_mode="shuffled",
    )
    naive_rows = [_score(group, (0.0,) * AGENTS) for group in groups]
    paired = {
        "macl_minus_fixed_balanced_mse": (
            macl_rows, fixed_rows, "balanced_test_mse"
        ),
        "macl_minus_fixed_task_a_retention_change": (
            macl_rows, fixed_rows, "task_a_retention_change"
        ),
        "macl_minus_fixed_task_b_mse": (
            macl_rows, fixed_rows, "task_b_test_mse_after_b"
        ),
        "macl_minus_self_signal_balanced_mse": (
            macl_rows, self_rows, "balanced_test_mse"
        ),
        "macl_minus_self_signal_task_a_retention_change": (
            macl_rows, self_rows, "task_a_retention_change"
        ),
        "macl_minus_shuffled_balanced_mse": (
            macl_rows, shuffled_rows, "balanced_test_mse"
        ),
        "macl_minus_naive_task_a_retention_change": (
            macl_rows, naive_rows, "task_a_retention_change"
        ),
    }

    def summarize(rows):
        keys = rows[0].keys()
        result = {}
        for key in keys:
            values = [row[key] for row in rows]
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "ci95_bootstrap": _bootstrap_mean_ci(values),
            }
        return result

    pair_summary = {}
    for key, (left_rows, right_rows, metric) in paired.items():
        values = [
            left[metric] - right[metric]
            for left, right in zip(left_rows, right_rows)
        ]
        pair_summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "ci95_bootstrap": _bootstrap_mean_ci(values),
            "macl_better_seed_count": sum(value < 0 for value in values),
            "seed_count": len(values),
        }
    return {
        "naive": summarize(naive_rows),
        "fixed_ewc": summarize(fixed_rows),
        "macl_partner_signal": summarize(macl_rows),
        "macl_self_signal_ablation": summarize(self_rows),
        "macl_shuffled_partner_signal_ablation": summarize(shuffled_rows),
        "paired_differences": pair_summary,
        "per_test_seed": [
            {
                "seed": seed,
                "naive": naive,
                "fixed_ewc": fixed,
                "macl_partner_signal": macl,
                "macl_self_signal": self_signal,
                "macl_shuffled_signal": shuffled,
            }
            for seed, naive, fixed, macl, self_signal, shuffled in zip(
                TEST_SEEDS, naive_rows, fixed_rows, macl_rows, self_rows, shuffled_rows
            )
        ],
    }


def _write_markdown(report):
    out = RESULTS / "CONTROLLED_MACL_STUDY.md"
    test = report["test_results"]

    def cell(method, metric):
        value = test[method][metric]
        return (
            f"{value['mean']:.5f} +/- {value['std']:.5f} "
            f"[{value['ci95_bootstrap'][0]:.5f}, "
            f"{value['ci95_bootstrap'][1]:.5f}]"
        )

    paired = test["paired_differences"]["macl_minus_fixed_balanced_mse"]
    lines = [
        "# Controlled synthetic MACL stress test",
        "",
        "## Protocol",
        "",
        "A two-agent linear-regression continual-learning diagnostic uses "
        f"{len(VALIDATION_SEEDS)} validation seeds for hyperparameter selection "
        f"and {len(TEST_SEEDS)} separate test seeds. Agents learn Task A, then "
        "Task B with varying but correlated task conflict. All methods use "
        f"identical data, initialization, and {STEPS} gradient steps per task.",
        "",
        "Fixed EWC strength and MACL's low/high strength and signal scale are "
        "selected by minimum validation mean of Task-A and Task-B test MSE. "
        f"Both arms receive {len(FIXED_LAMBDAS)} validation candidates. "
        "The partner-signal MACL arm is compared to fixed EWC, naive learning, "
        "a self-signal ablation, and a shuffled-partner-signal ablation. "
        "The reported 95% intervals are bootstrap intervals over test seeds.",
        "",
        f"Selected fixed EWC lambda: `{report['hyperparameters']['fixed_ewc_lambda']}`.",
        f"Selected MACL parameters: low=`{report['hyperparameters']['macl_low_lambda']}`, "
        f"high=`{report['hyperparameters']['macl_high_lambda']}`, "
        f"signal scale=`{report['hyperparameters']['macl_signal_scale']}`.",
        "",
        "## Held-out test results",
        "",
        "Lower is better for all losses. Retention change is Task-A test MSE "
        "after B minus after A; a smaller increase indicates less forgetting.",
        "",
        "| Method | Balanced MSE | Task-A retention change | Task-A MSE after B | Task-B MSE after B |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, title in (
        ("naive", "Naive"),
        ("fixed_ewc", "Validation-tuned fixed EWC"),
        ("macl_partner_signal", "MACL (partner signal)"),
        ("macl_self_signal_ablation", "MACL self-signal ablation"),
        ("macl_shuffled_partner_signal_ablation", "MACL shuffled-signal ablation"),
    ):
        lines.append(
            f"| {title} | "
            f"{cell(method, 'balanced_test_mse')} | "
            f"{cell(method, 'task_a_retention_change')} | "
            f"{cell(method, 'task_a_test_mse_after_b')} | "
            f"{cell(method, 'task_b_test_mse_after_b')} |"
        )
    lines.extend([
        "",
        "Paired MACL-minus-fixed-EWC balanced test MSE: "
        f"{paired['mean']:.6f} +/- {paired['std']:.6f}; bootstrap 95% CI "
        f"[{paired['ci95_bootstrap'][0]:.6f}, "
        f"{paired['ci95_bootstrap'][1]:.6f}]. MACL has lower balanced MSE on "
        f"{paired['macl_better_seed_count']}/{paired['seed_count']} test seeds.",
        "",
        "## Verdict for this benchmark",
        "",
        "**MACL does not outperform validation-tuned fixed EWC on the primary "
        "balanced metric in this study.** The 95% paired interval for "
        "MACL-minus-EWC balanced MSE is above zero, favoring fixed EWC. MACL "
        "does reduce Task-A forgetting relative to fixed EWC, but this comes "
        "with worse Task-B error. The partner-signal method also does not "
        "outperform the self-signal ablation, so this study does not isolate "
        "a benefit from multi-agent communication.",
        "",
        "",
        "The partner-signal arm is compared directly to both its own-signal "
        "ablation and a shuffled-partner-signal ablation. Lower paired "
        "differences favor partner communication:",
    ])
    for key in (
        "macl_minus_self_signal_balanced_mse",
        "macl_minus_self_signal_task_a_retention_change",
        "macl_minus_shuffled_balanced_mse",
        "macl_minus_fixed_task_a_retention_change",
        "macl_minus_fixed_task_b_mse",
    ):
        value = test["paired_differences"][key]
        lines.append(
            f"- `{key}`: {value['mean']:.6f}, bootstrap 95% CI "
            f"[{value['ci95_bootstrap'][0]:.6f}, "
            f"{value['ci95_bootstrap'][1]:.6f}], MACL is lower on "
            f"{value['macl_better_seed_count']}/{value['seed_count']} seeds."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This synthetic experiment tests a simplified adaptive-EWC mechanism, "
        "not the full MACL algorithm, language models, or negotiation. A result "
        "in this benchmark cannot prove general MACL superiority. A MACL win "
        "must also preserve or improve Task-B performance, outperform both "
        "self-signal and shuffled-signal ablations, and be replicated on the "
        "disjoint-prompt LLM study before supporting a broader claim.",
        "",
        "Machine-readable full results: [controlled_macl_study.json]"
        "(./controlled_macl_study.json).",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    validation = [_prepare_group(seed) for seed in VALIDATION_SEEDS]
    test = [_prepare_group(seed) for seed in TEST_SEEDS]
    hyperparameters = _select_hyperparameters(validation)
    report = {
        "protocol": {
            "type": "synthetic controlled diagnostic",
            "validation_seeds": list(VALIDATION_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "agents_per_seed": AGENTS,
            "dimensions": DIMENSIONS,
            "train_examples_per_task_agent": TRAIN_EXAMPLES,
            "test_examples_per_task_agent": TEST_EXAMPLES,
            "gradient_steps_per_task": STEPS,
            "metric": "equal-weight mean of Task-A and Task-B held-out MSE",
            "task_a_retention": "Task-A held-out MSE after B minus after A",
            "limitations": [
                "Synthetic linear regression; not LLM training or negotiation.",
                "Adaptive EWC signal is a simplified prototype, not a verified MACL reproduction.",
                "Hyperparameters are selected on validation seeds, not test seeds.",
            ],
        },
        "hyperparameters": hyperparameters,
    }
    report["test_results"] = _summarize_test(test, hyperparameters)
    json_path = RESULTS / "controlled_macl_study.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = _write_markdown(report)
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")
    for method in (
        "naive",
        "fixed_ewc",
        "macl_partner_signal",
        "macl_shuffled_partner_signal_ablation",
    ):
        score = report["test_results"][method]["balanced_test_mse"]
        print(f"{method}: balanced test MSE {score['mean']:.5f}")
    diff = report["test_results"]["paired_differences"][
        "macl_minus_fixed_balanced_mse"
    ]
    print(
        "MACL - fixed EWC: "
        f"{diff['mean']:.6f}, 95% bootstrap CI {diff['ci95_bootstrap']}"
    )


if __name__ == "__main__":
    main()
