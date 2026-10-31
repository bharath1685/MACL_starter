"""Held-out evaluation for the tabular Q-learning baseline.

This deliberately trains a fresh table on training episodes, then evaluates
greedily on independently generated fixed scenarios. It avoids relying on
repr-serialized Q-table keys from the previous pilot script.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from interactive_negotiation_env import NegotiationEnv, Scenario


def key(obs):
    offer = obs["current_offer_for_self"]
    return (
        obs["agent"], tuple(obs["stocks"]), tuple(obs["private_values"]),
        bool(obs["offer_exists"]), bool(obs["offer_from_other_agent"]),
        tuple(offer) if offer is not None else (-1, -1, -1),
        min(obs["turns_remaining"], 5),
    )


def actions(obs):
    stocks = obs["stocks"]
    out = [
        {"type": "offer", "allocation": [x, y, z]}
        for x in range(stocks[0] + 1)
        for y in range(stocks[1] + 1)
        for z in range(stocks[2] + 1)
    ]
    if obs["offer_from_other_agent"]:
        out.extend([{"type": "accept"}, {"type": "reject"}, {"type": "disagree"}])
    elif obs["offer_exists"]:
        out.extend([{"type": "reject"}, {"type": "disagree"}])
    else:
        out.append({"type": "disagree"})
    return out


def akey(a):
    return a["type"], tuple(a.get("allocation", ()))


def scenario(rng):
    stocks = tuple(rng.randint(1, 3) for _ in range(3))
    values = tuple(tuple(rng.randint(1, 5) for _ in range(3)) for _ in range(2))
    return Scenario(stocks, values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="outputs/tabular_q_heldout_eval.json")
    args = ap.parse_args()
    if args.train_episodes < 1 or args.eval_episodes < 1:
        ap.error("episode counts must be positive")

    rng = random.Random(args.seed)
    q = defaultdict(float)
    alpha, eps_start, eps_end = 0.15, 0.30, 0.03

    # Train from scratch, keeping Q table in memory for evaluation.
    for ep in range(args.train_episodes):
        sc = scenario(rng)
        env = NegotiationEnv(sc, max_turns=8, gap_weight=0.25)
        env.reset(seed=args.seed + ep)
        traj = []
        eps = eps_end + (eps_start - eps_end) * (1 - ep / args.train_episodes)
        while not env.done:
            agent = env.current_player
            obs = env.observe(agent)
            aa = actions(obs)
            kk = [akey(a) for a in aa]
            state = key(obs)
            if rng.random() < eps:
                i = rng.randrange(len(aa))
            else:
                vals = [q[(state, a)] for a in kk]
                best = max(vals)
                i = rng.choice([j for j, v in enumerate(vals) if v == best])
            traj.append((state, kk[i], agent))
            env.step(agent, aa[i])
        for state, act, agent in reversed(traj):
            target = env.rewards[agent]
            q[(state, act)] += alpha * (target - q[(state, act)])

    # Independent fixed held-out scenarios and greedy policy.
    eval_rng = random.Random(args.seed + 10_000_000)
    outcomes, rewards, turns, invalid = [], [], [], 0
    for ep in range(args.eval_episodes):
        sc = scenario(eval_rng)
        env = NegotiationEnv(sc, max_turns=8, gap_weight=0.25)
        env.reset(seed=args.seed + 20_000_000 + ep)
        while not env.done:
            agent = env.current_player
            obs = env.observe(agent)
            aa = actions(obs)
            state = key(obs)
            vals = [q[(state, akey(a))] for a in aa]
            best = max(vals)
            idx = [i for i, v in enumerate(vals) if v == best]
            # Deterministic tie-breaking for reproducible evaluation.
            action = aa[idx[0]]
            try:
                env.step(agent, action)
            except (ValueError, RuntimeError):
                invalid += 1
                break
        outcomes.append(env.outcome)
        rewards.append(env.rewards)
        turns.append(env.turns)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose another --output")
    deals = sum(x == "deal" for x in outcomes)
    report = {
        "training_episodes": args.train_episodes,
        "heldout_episodes": args.eval_episodes,
        "seed": args.seed,
        "heldout_deal_rate": deals / args.eval_episodes,
        "heldout_mean_reward_agent0": sum(r[0] for r in rewards) / args.eval_episodes,
        "heldout_mean_reward_agent1": sum(r[1] for r in rewards) / args.eval_episodes,
        "heldout_mean_turns": sum(turns) / args.eval_episodes,
        "invalid_action_or_transition_count": invalid,
        "outcome_counts": {x: outcomes.count(x) for x in sorted(set(outcomes))},
        "caveat": "Single-seed tabular baseline; evaluate multiple seeds before scientific conclusions.",
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved held-out report: {output}")


if __name__ == "__main__":
    main()
