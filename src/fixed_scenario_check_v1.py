import random
from src.interactive_negotiation_env import Scenario
from src.continual_reward_env import ContinualRewardEnv

def make_scenarios(n=100, seed=2026):
    rng = random.Random(seed)
    return [
        Scenario(
            stocks=tuple(rng.randint(1, 3) for _ in range(3)),
            values=tuple(
                tuple(rng.randint(0, 5) for _ in range(3))
                for _ in range(2)
            ),
        )
        for _ in range(n)
    ]

def evaluate(scenarios, objective):
    rewards, deals = [], 0
    for scenario in scenarios:
        env = ContinualRewardEnv(scenario, objective=objective, max_turns=6)
        obs = env.reset()
        done = False
        while not done:
            agent = env.current_player
            o = obs[agent]
            if o["offer_from_other_agent"]:
                offer = o["current_offer_for_self"]
                own_value = sum(v*x for v, x in zip(o["private_values"], offer))
                action = {"type": "accept"} if own_value > 0 else {"type": "reject"}
            else:
                action = {
                    "type": "offer",
                    "allocation": tuple(s // 2 for s in o["stocks"]),
                }
            a, b, result, done = env.step(agent, action)
            obs = (a, b)
        rewards.append(sum(result) / 2)
        deals += env.outcome == "deal"
    return round(sum(rewards) / len(rewards), 3), round(deals / len(rewards), 3)

scenarios = make_scenarios()
print("Fixed scenario count:", len(scenarios))
print("Self-interest objective:", evaluate(scenarios, "self_interest"))
print("Fairness objective:", evaluate(scenarios, "fairness"))
print("PASS: both objectives evaluated on identical scenarios")
