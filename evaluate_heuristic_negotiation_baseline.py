"""Evaluate a transparent alternating-offer heuristic on held-out scenarios."""
import argparse
import json
import random
from pathlib import Path

from interactive_negotiation_env import NegotiationEnv, Scenario


def make_scenario(rng):
    stocks = tuple(rng.randint(1, 3) for _ in range(3))
    values = tuple(tuple(rng.randint(1, 5) for _ in range(3)) for _ in range(2))
    return Scenario(stocks, values)


def own_payoff_from_observation(obs, own_allocation):
    vals = obs["private_values"]
    return sum(own_allocation[i] * vals[i] for i in range(3))


def heuristic_action(obs, accept_fraction=0.40):
    stocks = tuple(obs["stocks"])
    agent = obs["agent"]
    # An offer is always expressed as the number of items allocated to agent 0.
    if obs["offer_from_other_agent"]:
        offer = tuple(obs["current_offer_for_self"])
        max_own = sum(obs["stocks"][i] * obs["private_values"][i] for i in range(3))
        if own_payoff_from_observation(obs, offer) >= accept_fraction * max_own:
            return {"type": "accept"}
    # Both players propose the same balanced split; odd items are assigned
    # to the proposer where possible, while action coordinates remain agent-0 shares.
    allocation = []
    for stock in stocks:
        half = stock // 2
        if agent == 0:
            allocation.append((stock + 1) // 2)
        else:
            allocation.append(half)
    return {"type": "offer", "allocation": allocation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="outputs/heuristic_negotiation_heldout_eval.json")
    args = ap.parse_args()
    if args.episodes < 1:
        ap.error("--episodes must be positive")
    rng = random.Random(args.seed + 10_000_000)
    outcomes, rewards, turns, payoff_gaps = [], [], [], []
    for i in range(args.episodes):
        env = NegotiationEnv(make_scenario(rng), max_turns=8, gap_weight=0.25)
        env.reset(seed=args.seed + 20_000_000 + i)
        while not env.done:
            agent = env.current_player
            obs = env.observe(agent)
            action = heuristic_action(obs)
            env.step(agent, action)
        outcomes.append(env.outcome)
        rewards.append(tuple(env.rewards))
        turns.append(env.turns)
        if env.outcome == "deal":
            sc = env.scenario
            offer = env.outcome_allocation if hasattr(env, "outcome_allocation") else None
            # Environment's terminal rewards already reflect the configured gap penalty.
            if offer is not None:
                pass
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose another --output")
    report = {
        "method": "balanced alternating-offer heuristic",
        "episodes": args.episodes,
        "seed": args.seed,
        "deal_rate": outcomes.count("deal") / args.episodes,
        "mean_reward_agent0": sum(x[0] for x in rewards) / args.episodes,
        "mean_reward_agent1": sum(x[1] for x in rewards) / args.episodes,
        "mean_turns": sum(turns) / args.episodes,
        "outcome_counts": {x: outcomes.count(x) for x in sorted(set(outcomes))},
        "caveat": "Rule-based reference baseline, not a learned policy; compare on the same held-out scenarios as other methods.",
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
