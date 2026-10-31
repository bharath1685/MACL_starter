"""Train a simple tabular Q-learning baseline in the MACL negotiation environment.

Run from MACL_starter root:
  PYTHONPATH=src python src/train_tabular_negotiation_baseline.py --episodes 5000 --seed 17

This is a first learned-policy baseline, not a transformer or MACL result.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from interactive_negotiation_env import NegotiationEnv, Scenario


def state_key(obs):
    # Small, deliberately coarse state abstraction for a tractable baseline.
    stocks = tuple(obs["stocks"])
    values = tuple(obs["private_values"])
    offer = obs["current_offer_for_self"]
    if offer is None:
        offer = (-1, -1, -1)
    return (
        obs["agent"], stocks, values, bool(obs["offer_exists"]),
        bool(obs["offer_from_other_agent"]), tuple(offer),
        min(obs["turns_remaining"], 5),
    )


def legal_actions(obs):
    # The environment's offer always specifies agent 0's allocation.
    stocks = obs["stocks"]
    actions = [
        {"type": "offer", "allocation": [x, y, z]}
        for x in range(stocks[0] + 1)
        for y in range(stocks[1] + 1)
        for z in range(stocks[2] + 1)
    ]
    if obs["offer_from_other_agent"]:
        actions += [{"type": "accept"}, {"type": "reject"}, {"type": "disagree"}]
    elif obs["offer_exists"]:
        actions += [{"type": "reject"}, {"type": "disagree"}]
    else:
        actions += [{"type": "disagree"}]
    return actions


def action_key(action):
    return (action["type"], tuple(action.get("allocation", ())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="outputs/tabular_q_baseline.json")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    q = defaultdict(float)
    alpha, gamma = 0.15, 0.95
    eps_start, eps_end = 0.30, 0.03
    recent_rewards, recent_deals = [], []

    for ep in range(args.episodes):
        stocks = tuple(rng.randint(1, 3) for _ in range(3))
        values = tuple(tuple(rng.randint(1, 5) for _ in range(3)) for _ in range(2))
        env = NegotiationEnv(
            Scenario(stocks=stocks, values=values),
            max_turns=8,
            gap_weight=0.25,
        )
        env.reset(seed=args.seed + ep)
        eps = eps_end + (eps_start - eps_end) * (1 - ep / args.episodes)
        transitions = []

        while not env.done:
            agent = env.current_player
            obs = env.observe(agent)
            s = state_key(obs)
            actions = legal_actions(obs)
            keys = [action_key(a) for a in actions]
            if rng.random() < eps:
                idx = rng.randrange(len(actions))
            else:
                vals = [q[(s, ak)] for ak in keys]
                best = max(vals)
                candidates = [i for i, val in enumerate(vals) if val == best]
                idx = rng.choice(candidates)
            action = actions[idx]
            transitions.append((s, keys[idx], agent))
            env.step(agent, action)

        # Team-independent terminal reward is assigned to each agent's own turns.
        # Update each agent's state-action values with its own terminal reward.
        for s, ak, agent in reversed(transitions):
            target = env.rewards[agent]
            q[(s, ak)] += alpha * (target - q[(s, ak)])

        recent_rewards.append(sum(env.rewards) / 2)
        recent_deals.append(int(env.outcome == "deal"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose another --output")
    report = {
        "method": "independent tabular Monte-Carlo terminal-return baseline",
        "episodes": args.episodes,
        "seed": args.seed,
        "last_100_mean_agent_reward": sum(recent_rewards[-100:]) / min(100, len(recent_rewards)),
        "last_100_deal_rate": sum(recent_deals[-100:]) / min(100, len(recent_deals)),
        "q_state_action_entries": len(q),
        "caveat": "Exploratory RL baseline; evaluate on a separate fixed held-out scenario set before drawing conclusions.",
    }
    # Save learned table as JSON with serialized keys for reproducibility.
    report["q_table"] = [
        {"state": repr(s), "action": repr(a), "value": v}
        for (s, a), v in q.items()
    ]
    out.write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "q_table"}, indent=2))
    print(f"Saved learned Q-table and report: {out}")


if __name__ == "__main__":
    main()
