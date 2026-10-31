#!/usr/bin/env python3
"""Fixed-compute synthetic scaling study for multi-agent forgetting signals.

This is a linear-regression mechanism test, not an LLM negotiation experiment
or a reproduction of the MACL paper. Each team gets a fixed number of optimizer
updates regardless of team size. Hyperparameters are tuned on validation
seeds, then all arms are compared on separate held-out seeds.
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEAM_SIZES = (1, 2, 4, 8)
VALIDATION_SEEDS = tuple(range(3000, 3024))
TEST_SEEDS = tuple(range(4000, 4100))
TOTAL_UPDATES_PER_TASK = 320
DIMENSIONS = 8
TRAIN_EXAMPLES = 128
TEST_EXAMPLES = 512
LEARNING_RATE = 0.035
NOISE = 0.08
FIXED_LAMBDAS = tuple([0.0, *np.geomspace(0.01, 12.0, 107).tolist()])
ADAPTIVE_LOW = (0.0, 0.1, 0.3)
ADAPTIVE_HIGH = (0.3, 0.7, 1.5, 3.0, 6.0, 12.0)
ADAPTIVE_SCALE = (0.01, 0.1, 0.5, 1.0, 3.0, 10.0)
METHODS = (
    "naive",
    "fixed_ewc",
    "local_adaptive",
    "peer_communication",
    "shuffled_communication",
)


def _steps_per_agent(team_size):
    if team_size not in TEAM_SIZES:
        raise ValueError(f"Unsupported team size: {team_size}")
    if TOTAL_UPDATES_PER_TASK % team_size:
        raise ValueError("Team size must divide the fixed update budget.")
    return TOTAL_UPDATES_PER_TASK // team_size


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def make_team(seed, team_size):
    """Create one team with a shared Task A and heterogeneous Task B tasks."""
    updates = _steps_per_agent(team_size)
    rng = np.random.default_rng(seed)
    target_a = _unit(rng.normal(size=DIMENSIONS)) * math.sqrt(DIMENSIONS)
    agents = []

    for agent_index in range(team_size):
        direction = rng.normal(size=DIMENSIONS)
        direction -= (
            np.dot(direction, target_a) / np.dot(target_a, target_a)
        ) * target_a
        direction = _unit(direction) * math.sqrt(DIMENSIONS)
        severity = float(np.clip(rng.normal(1.0, 0.45), 0.15, 1.9))
        angle = severity * math.pi / 2
        target_b = math.cos(angle) * target_a + math.sin(angle) * direction

        data_rng = np.random.default_rng(seed * 1009 + agent_index * 7919 + 31)
        train_a = _make_data(data_rng, target_a, TRAIN_EXAMPLES)
        train_b = _make_data(data_rng, target_b, TRAIN_EXAMPLES)
        test_a = _make_data(data_rng, target_a, TEST_EXAMPLES)
        test_b = _make_data(data_rng, target_b, TEST_EXAMPLES)
        initial = rng.normal(scale=0.01, size=DIMENSIONS)
        theta_a = _fit(initial, train_a, updates)
        fisher = _estimate_fisher(theta_a, train_a)
        task_b_gradient = _gradient(theta_a, train_b)
        signal = float(np.mean(fisher * task_b_gradient**2))
        agents.append({
            "severity": severity,
            "theta_a": theta_a,
            "fisher": fisher,
            "signal": signal,
            "train_b": train_b,
            "test_a": test_a,
            "test_b": test_b,
            "a_after_a": _mse(theta_a, test_a),
        })

    return {"seed": seed, "team_size": team_size, "agents": agents}


def _make_data(rng, target, count):
    x = rng.normal(size=(count, DIMENSIONS))
    y = x @ target + rng.normal(scale=NOISE, size=count)
    return x, y


def _mse(theta, data):
    x, y = data
    residual = x @ theta - y
    return float(np.mean(residual**2))


def _gradient(theta, data):
    x, y = data
    return (2.0 / len(x)) * (x.T @ (x @ theta - y))


def _estimate_fisher(theta, data):
    x, y = data
    per_example_grad = 2.0 * x * (x @ theta - y)[:, None]
    fisher = np.mean(per_example_grad**2, axis=0)
    mean = float(np.mean(fisher))
    if mean > 0:
        fisher = fisher / mean
    return np.maximum(fisher, 1e-8)


def _fit(theta, data, steps, fisher=None, anchor=None, strength=0.0):
    x, y = data
    theta = theta.copy()
    for _ in range(steps):
        residual = x @ theta - y
        gradient = (2.0 / len(x)) * (x.T @ residual)
        if strength:
            gradient += 2.0 * strength * fisher * (theta - anchor)
        theta -= LEARNING_RATE * gradient
    return theta


def _adaptive_strength(signal, low, high, scale):
    signal = max(0.0, float(signal))
    return low + (high - low) * signal / (signal + scale)


def _received_signals(team, method, shuffled_team=None):
    signals = [agent["signal"] for agent in team["agents"]]
    n_agents = len(signals)
    if method == "peer_communication" and n_agents == 1:
        return signals
    if method == "shuffled_communication" and n_agents == 1:
        return [shuffled_team["agents"][0]["signal"]]
    if method == "peer_communication":
        return [
            statistics.mean(signals[:i] + signals[i + 1:])
            for i in range(n_agents)
        ]
    if method == "shuffled_communication":
        peers = [agent["signal"] for agent in shuffled_team["agents"]]
        return [
            statistics.mean(peers[:i] + peers[i + 1:])
            if n_agents > 1
            else peers[0]
            for i in range(n_agents)
        ]
    if method == "local_adaptive":
        return signals
    return [0.0] * n_agents


def _evaluate_team(team, method, settings, steps, shuffled_team=None):
    signals = _received_signals(team, method, shuffled_team)
    rows = []
    for agent, signal in zip(team["agents"], signals):
        if method == "naive":
            strength = 0.0
        elif method == "fixed_ewc":
            strength = settings["fixed_lambda"]
        else:
            strength = _adaptive_strength(
                signal,
                settings["low_lambda"],
                settings["high_lambda"],
                settings["signal_scale"],
            )
        theta_b = _fit(
            agent["theta_a"],
            agent["train_b"],
            steps,
            fisher=agent["fisher"],
            anchor=agent["theta_a"],
            strength=strength,
        )
        rows.append({
            "task_a_test_mse": _mse(theta_b, agent["test_a"]),
            "task_b_test_mse": _mse(theta_b, agent["test_b"]),
            "task_a_retention_change": (
                _mse(theta_b, agent["test_a"]) - agent["a_after_a"]
            ),
            "strength": strength,
        })
    return {
        "balanced_test_mse": statistics.mean(
            (row["task_a_test_mse"] + row["task_b_test_mse"]) / 2
            for row in rows
        ),
        "task_a_test_mse_after_b": statistics.mean(
            row["task_a_test_mse"] for row in rows
        ),
        "task_b_test_mse_after_b": statistics.mean(
            row["task_b_test_mse"] for row in rows
        ),
        "task_a_retention_change": statistics.mean(
            row["task_a_retention_change"] for row in rows
        ),
        "mean_consolidation_strength": statistics.mean(
            row["strength"] for row in rows
        ),
    }


def _candidate_settings(method):
    if method == "fixed_ewc":
        return [{"fixed_lambda": strength} for strength in FIXED_LAMBDAS]
    if method in ("local_adaptive", "peer_communication", "shuffled_communication"):
        return [
            {
                "low_lambda": low,
                "high_lambda": high,
                "signal_scale": scale,
            }
            for low in ADAPTIVE_LOW
            for high in ADAPTIVE_HIGH
            if high >= low
            for scale in ADAPTIVE_SCALE
        ]
    return [{}]


def _balanced_validation_score(teams, method, settings, steps, shuffled_teams):
    scores = []
    for index, team in enumerate(teams):
        shuffled = (
            shuffled_teams[(index + 1) % len(shuffled_teams)]
            if method == "shuffled_communication"
            else None
        )
        score = _evaluate_team(team, method, settings, steps, shuffled)
        scores.append(score["balanced_test_mse"])
    return float(np.mean(scores))


def tune(validation_teams, team_size):
    steps = _steps_per_agent(team_size)
    selected = {}
    for method in METHODS:
        if method == "naive":
            selected[method] = {}
            continue
        candidates = _candidate_settings(method)
        scores = [
            _balanced_validation_score(
                validation_teams,
                method,
                candidate,
                steps,
                validation_teams,
            )
            for candidate in candidates
        ]
        selected[method] = candidates[int(np.argmin(scores))]
        selected[method]["validation_balanced_mse"] = float(min(scores))
        selected[method]["candidate_count"] = len(candidates)
    return selected


def _bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[index].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def evaluate_test(test_teams, selected, team_size):
    steps = _steps_per_agent(team_size)
    per_method = {}
    for method in METHODS:
        per_method[method] = []
        for index, team in enumerate(test_teams):
            shuffled = (
                test_teams[(index + 1) % len(test_teams)]
                if method == "shuffled_communication"
                else None
            )
            per_method[method].append(
                _evaluate_team(
                    team, method, selected[method], steps, shuffled
                )
            )

    summaries = {}
    for method, rows in per_method.items():
        summaries[method] = {}
        for metric in rows[0]:
            values = [row[metric] for row in rows]
            summaries[method][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "ci95_bootstrap": _bootstrap_ci(values, 7100 + team_size),
            }

    comparisons = {}
    for comparator in ("fixed_ewc", "local_adaptive", "shuffled_communication"):
        deltas = [
            per_method["peer_communication"][i]["balanced_test_mse"]
            - per_method[comparator][i]["balanced_test_mse"]
            for i in range(len(test_teams))
        ]
        comparisons[f"peer_minus_{comparator}_balanced_mse"] = {
            "mean": float(np.mean(deltas)),
            "std": float(np.std(deltas, ddof=1)),
            "ci95_bootstrap": _bootstrap_ci(deltas, 8100 + team_size),
            "peer_better_seed_count": sum(delta < 0 for delta in deltas),
            "seed_count": len(deltas),
        }
    return {
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "per_test_seed": [
            {
                "seed": team["seed"],
                **{
                    method: per_method[method][i]
                    for method in METHODS
                },
            }
            for i, team in enumerate(test_teams)
        ],
    }


def _write_markdown(report):
    path = RESULTS / "MULTIAGENT_SCALING_STUDY.md"
    lines = [
        "# Multi-agent communication scaling study",
        "",
        "## Protocol",
        "",
        "This synthetic linear-regression study tests whether exchanging "
        "estimated Task-A interference across a team improves retention. "
        "Team sizes are 1, 2, 4, and 8. Each team receives exactly "
        f"{TOTAL_UPDATES_PER_TASK} optimizer updates per task; validation and "
        f"test sets use {len(VALIDATION_SEEDS)} and {len(TEST_SEEDS)} "
        "independent seeds. Fixed EWC and each adaptive arm receive equal "
        "108-candidate validation searches. All methods share the same "
        "generated tasks and test seeds at each team size.",
        "",
        "All scores are held-out means over teams; lower is better. "
        "Task-A retention change is Task-A test MSE after Task B minus after "
        "Task A. The 95% intervals are bootstrap intervals over test seeds.",
        "",
        "## Results by team size",
        "",
        "| Agents | Method | Balanced MSE | Task-A retention change | Task-B MSE |",
        "|---:|---|---:|---:|---:|",
    ]
    for size in TEAM_SIZES:
        result = report["team_sizes"][str(size)]["test_results"]
        for method in METHODS:
            metrics = result["summaries"][method]
            lines.append(
                f"| {size} | {method} | "
                f"{metrics['balanced_test_mse']['mean']:.4f} "
                f"[{metrics['balanced_test_mse']['ci95_bootstrap'][0]:.4f}, "
                f"{metrics['balanced_test_mse']['ci95_bootstrap'][1]:.4f}] | "
                f"{metrics['task_a_retention_change']['mean']:.4f} "
                f"[{metrics['task_a_retention_change']['ci95_bootstrap'][0]:.4f}, "
                f"{metrics['task_a_retention_change']['ci95_bootstrap'][1]:.4f}] | "
                f"{metrics['task_b_test_mse_after_b']['mean']:.4f} |"
            )
    lines.extend([
        "",
        "## Paired peer-communication comparisons",
        "",
        "A negative peer-minus-control value favors peer communication:",
        "",
        "| Agents | Comparator | Balanced-MSE difference | Bootstrap 95% CI | Peer wins |",
        "|---:|---|---:|---:|---:|",
    ])
    for size in TEAM_SIZES:
        comparisons = report["team_sizes"][str(size)]["test_results"][
            "paired_comparisons"
        ]
        for comparator in ("fixed_ewc", "local_adaptive", "shuffled_communication"):
            delta = comparisons[f"peer_minus_{comparator}_balanced_mse"]
            lines.append(
                f"| {size} | {comparator} | {delta['mean']:.5f} | "
                f"[{delta['ci95_bootstrap'][0]:.5f}, "
                f"{delta['ci95_bootstrap'][1]:.5f}] | "
                f"{delta['peer_better_seed_count']}/{delta['seed_count']} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This isolates a small adaptive-consolidation signal in a synthetic "
        "problem; it cannot establish MACL performance on LLMs or negotiation. "
        "Increasing agent count changes the distribution of agent-specific "
        "tasks while keeping the total number of optimizer updates fixed. "
        "The one-agent peer-communication arm has no peers and uses its local "
        "signal as a documented fallback. Treat trends as exploratory; a "
        "favorable result needs replication on disjoint negotiation tasks.",
        "",
        "Machine-readable results: [multiagent_scaling_study.json]"
        "(./multiagent_scaling_study.json).",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_study():
    report = {
        "protocol": {
            "type": "synthetic fixed-compute scaling diagnostic",
            "team_sizes": list(TEAM_SIZES),
            "validation_seeds": list(VALIDATION_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "total_optimizer_updates_per_team_per_task": TOTAL_UPDATES_PER_TASK,
            "per_agent_optimizer_updates": {
                str(size): _steps_per_agent(size) for size in TEAM_SIZES
            },
            "fixed_lambda_candidates": len(FIXED_LAMBDAS),
            "adaptive_candidates": len(_candidate_settings("peer_communication")),
            "communication_signal": (
                "Fisher-weighted squared Task-B gradient at the Task-A solution"
            ),
            "limitations": [
                "Synthetic linear regression; not LLM training or negotiation.",
                "Communication signal and adaptive-lambda mapping are prototype choices.",
                "Agent counts exceed two only in this synthetic diagnostic.",
            ],
        },
        "team_sizes": {},
    }
    for size in TEAM_SIZES:
        validation = [
            make_team(seed, size) for seed in VALIDATION_SEEDS
        ]
        test = [make_team(seed, size) for seed in TEST_SEEDS]
        selected = tune(validation, size)
        test_results = evaluate_test(test, selected, size)
        report["team_sizes"][str(size)] = {
            "selected_hyperparameters": selected,
            "test_results": test_results,
        }
        print(
            f"agents={size}: " + ", ".join(
                f"{method}={test_results['summaries'][method]['balanced_test_mse']['mean']:.4f}"
                for method in METHODS
            ),
            flush=True,
        )
    return report


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run_study()
    json_path = RESULTS / "multiagent_scaling_study.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path = _write_markdown(report)
    print(f"Saved {json_path}")
    print(f"Saved {markdown_path}")


if __name__ == "__main__":
    main()
