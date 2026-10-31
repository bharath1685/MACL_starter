"""Version 2: synthetic continual-learning diagnostic.

This is NOT an LLM/RL experiment and does not validate MACL.
The adaptive lambda formula is a documented toy design choice.

EWC uses online cumulative diagonal importance:
F_online <- decay * F_online + F_current
with the consolidation anchor set after each task.
"""

from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = (7, 17, 27, 37, 47)
AGENTS = 2
STEPS = 180
LR = 0.035
FISHER_DECAY = 1.0
FIXED_LAMBDA = 0.7
LAMBDA_BASE = 0.2
LAMBDA_SCALE = 1.0
LAMBDA_MIN = 0.2
LAMBDA_MAX = 1.5
SIGNAL_SCALE = 9.0


def adaptive_lambda(signal):
    """Smoothly map partner feedback to bounded EWC strength.

    This is an engineering choice for the toy diagnostic, not a
    numerical formula claimed to come from the MACL paper.
    """
    import numpy as np
    normalized = max(0.0, float(signal)) / (
        SIGNAL_SCALE + max(0.0, float(signal))
    )
    return float(np.clip(
        LAMBDA_BASE + LAMBDA_SCALE * normalized,
        LAMBDA_MIN,
        LAMBDA_MAX,
    ))

TASKS = {
    "A_self_interest": np.array([2.0, -1.0]),
    "B_fairness": np.array([-1.2, 1.8]),
}
SEQUENCE = ("A_self_interest", "B_fairness", "A_self_interest")
METHODS = ("naive", "fixed_ewc", "macl_toy")


def make_data(rng, target, n, noise=0.08):
    x = rng.normal(size=(n, len(target)))
    y = x @ target + rng.normal(scale=noise, size=n)
    return x, y


def task_data(seed):
    rng = np.random.default_rng(seed)
    train, test = {}, {}
    for name, target in TASKS.items():
        test[name] = make_data(rng, target, 1000)
        train[name] = [
            make_data(rng, target, 160) for _ in range(AGENTS)
        ]
    return train, test


def mse(theta, data):
    x, y = data
    return float(np.mean((x @ theta - y) ** 2))


def gradient(theta, data):
    x, y = data
    residual = x @ theta - y
    return (2.0 / len(x)) * (x.T @ residual)


def empirical_fisher_diagonal(theta, data):
    """Mean squared per-example gradients: empirical diagonal Fisher proxy."""
    x, y = data
    residual = x @ theta - y
    per_example = 2.0 * x * residual[:, None]
    return np.maximum(np.mean(per_example ** 2, axis=0), 1e-8)


def fit(theta, data, fisher, anchor, lam, steps=STEPS):
    theta = theta.copy()
    for _ in range(steps):
        grad = gradient(theta, data)
        if lam > 0:
            grad += 2.0 * lam * fisher * (theta - anchor)
        theta -= LR * grad
    return theta


def evaluate(agents, test):
    return {
        task: float(np.mean([-mse(a, test[task]) for a in agents]))
        for task in TASKS
    }


def proposed_unregularized_drift(theta, data, anchor, steps=STEPS):
    """Estimate how far the next task would move an agent without EWC."""
    proposed = fit(
        theta, data,
        fisher=np.zeros_like(theta),
        anchor=anchor,
        lam=0.0,
        steps=steps,
    )
    return proposed - anchor


def run_method(method, seed):
    train, test = task_data(seed)
    rng = np.random.default_rng(seed + 10000)

    agents = [rng.normal(scale=0.01, size=2) for _ in range(AGENTS)]
    anchors = [a.copy() for a in agents]
    fishers = [np.zeros(2, dtype=float) for _ in range(AGENTS)]

    matrix = np.full((3, 2), np.nan)
    logs = []

    for stage, task in enumerate(SEQUENCE):
        start = [a.copy() for a in agents]

        # Compute both agents' proposed drifts before either agent is updated.
        proposed_drifts = [
            proposed_unregularized_drift(
                start[i], train[task][i], anchors[i]
            )
            for i in range(AGENTS)
        ]

        updated = []
        for i in range(AGENTS):
            lam, signal = 0.0, 0.0

            if stage > 0 and method == "fixed_ewc":
                lam = FIXED_LAMBDA

            elif stage > 0 and method == "macl_toy":
                partner = 1 - i
                pf = fishers[partner]
                drift = proposed_drifts[partner]
                signal = float(
                    np.sum(pf * drift**2) / (np.sum(pf) + 1e-12)
                )
                lam = adaptive_lambda(signal)

            updated.append(fit(
                start[i],
                train[task][i],
                fishers[i],
                anchors[i],
                lam,
            ))

            logs.append({
                "seed": seed,
                "method": method,
                "stage": stage + 1,
                "agent": i + 1,
                "task": task,
                "partner_proposed_drift_signal": signal,
                "lambda": lam,
            })

        agents = updated

        # Online EWC: retain cumulative importance rather than replacing it.
        for i in range(AGENTS):
            current_fisher = empirical_fisher_diagonal(
                agents[i], train[task][i]
            )
            fishers[i] = (
                FISHER_DECAY * fishers[i] + current_fisher
            )
            anchors[i] = agents[i].copy()

        scores = evaluate(agents, test)
        for j, task_name in enumerate(TASKS):
            matrix[stage, j] = scores[task_name]

    return matrix, logs


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    matrices = {}
    signal_rows = []
    matrix_rows = []

    for seed in SEEDS:
        for method in METHODS:
            matrix, logs = run_method(method, seed)
            matrices[(seed, method)] = matrix
            signal_rows.extend(logs)

            for stage, task in enumerate(SEQUENCE):
                matrix_rows.append({
                    "seed": seed,
                    "method": method,
                    "stage": stage + 1,
                    "trained_task": task,
                    "score_A": matrix[stage, 0],
                    "score_B": matrix[stage, 1],
                })

    # Stage-2 forgetting: A after learning B minus A immediately after A.
    # Final retention: report final scores separately because A is learned again
    # at Stage 3, so final A is not a clean measure of forgetting from Stage 1.
    summary = []
    for method in METHODS:
        a1 = [matrices[(s, method)][0, 0] for s in SEEDS]
        a2 = [matrices[(s, method)][1, 0] for s in SEEDS]
        a3 = [matrices[(s, method)][2, 0] for s in SEEDS]
        b2 = [matrices[(s, method)][1, 1] for s in SEEDS]
        b3 = [matrices[(s, method)][2, 1] for s in SEEDS]

        stage2_bwt, stage2_bwt_std = mean_std(
            [y - x for x, y in zip(a1, a2)]
        )
        final_a, final_a_std = mean_std(a3)
        final_b, final_b_std = mean_std(b3)
        stage1_a, stage1_a_std = mean_std(a1)
        stage2_a, stage2_a_std = mean_std(a2)
        stage2_b, stage2_b_std = mean_std(b2)

        summary.append({
            "method": method,
            "seed_count": len(SEEDS),
            "A_after_stage1_mean": stage1_a,
            "A_after_stage1_std": stage1_a_std,
            "A_after_stage2_mean": stage2_a,
            "A_after_stage2_std": stage2_a_std,
            "B_after_stage2_mean": stage2_b,
            "B_after_stage2_std": stage2_b_std,
            "A_stage2_change_mean": stage2_bwt,
            "A_stage2_change_std": stage2_bwt_std,
            "A_after_stage3_mean": final_a,
            "A_after_stage3_std": final_a_std,
            "B_after_stage3_mean": final_b,
            "B_after_stage3_std": final_b_std,
        })

    prefix = "toy_v2"
    matrix_path = OUT / f"{prefix}_reward_matrix.csv"
    signal_path = OUT / f"{prefix}_communication_log.csv"
    summary_path = OUT / f"{prefix}_summary.csv"
    plot_path = OUT / f"{prefix}_task_A_retention.png"
    metadata_path = OUT / f"{prefix}_metadata.json"

    write_csv(matrix_path, matrix_rows, list(matrix_rows[0].keys()))
    write_csv(signal_path, signal_rows, list(signal_rows[0].keys()))
    write_csv(summary_path, summary, list(summary[0].keys()))

    fig, ax = plt.subplots(figsize=(8, 5))
    stages = np.arange(1, 4)
    for method in METHODS:
        values = np.array([matrices[(s, method)][:, 0] for s in SEEDS])
        avg = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        ax.plot(stages, avg, marker="o", label=method)
        ax.fill_between(stages, avg - sd, avg + sd, alpha=0.15)
    ax.set_title("Synthetic diagnostic v2: Task A score")
    ax.set_xlabel("Training stage")
    ax.set_ylabel("Task A score (negative MSE; higher is better)")
    ax.set_xticks(stages)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    metadata = {
        "type": "synthetic diagnostic only",
        "seeds": list(SEEDS),
        "sequence": list(SEQUENCE),
        "methods": list(METHODS),
        "fisher": "mean squared per-example gradients",
        "ewc": "online cumulative diagonal importance",
        "fisher_decay": FISHER_DECAY,
        "adaptive_lambda": {
            "formula": "clip(base + scale * partner_proposed_drift_signal, min, max)",
            "base": LAMBDA_BASE,
            "scale": LAMBDA_SCALE,
            "signal_scale": SIGNAL_SCALE,
            "min": LAMBDA_MIN,
            "max": LAMBDA_MAX,
            "status": "toy design choice, not specified numerically by paper",
        },
        "warning": "Not an LLM/RL experiment; does not validate MACL.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Synthetic continual-learning diagnostic v2 completed.")
    print("Seeds:", SEEDS)
    print("\nTask A change after Stage 2 (negative means forgetting):")
    for row in summary:
        print(
            f"{row['method']:10s} "
            f"{row['A_stage2_change_mean']:.4f} "
            f"+/- {row['A_stage2_change_std']:.4f}"
        )

    print("\nFiles saved:")
    for path in (matrix_path, signal_path, summary_path, plot_path, metadata_path):
        print(path)

    print("\nLIMITATION: synthetic diagnostic only; not evidence of MACL effectiveness.")


if __name__ == "__main__":
    main()
