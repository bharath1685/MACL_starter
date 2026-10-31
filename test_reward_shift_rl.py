import random
from collections import defaultdict

from src.continual_reward_env import ContinualRewardEnv
from src.interactive_negotiation_env import Scenario

SEED = 17
random.seed(SEED)

# Small, reproducible scenario distribution.
def sample_scenario():
    stocks = tuple(random.randint(1, 3) for _ in range(3))
    values = tuple(
        tuple(random.randint(0, 5) for _ in range(3))
        for _ in range(2)
    )
    return Scenario(stocks=stocks, values=values)

def state_key(obs):
    offer = obs["current_offer_for_self"]
    return (
        obs["agent"],
        obs["private_values"],
        obs["stocks"],
        obs["offer_from_other_agent"],
        tuple(offer) if offer is not None else (-1, -1, -1),
        obs["turns_remaining"],
    )

def actions(obs):
    # Keep the initial experiment small: offer all, offer nothing,
    # or make a balanced offer; accept/reject if an offer is pending.
    stocks = obs["stocks"]
    candidates = [
        {"type": "offer", "allocation": tuple(stocks)},
        {"type": "offer", "allocation": (0, 0, 0)},
        {"type": "offer", "allocation": tuple(s // 2 for s in stocks)},
    ]
    if obs["offer_from_other_agent"]:
        return [{"type": "accept"}, {"type": "reject"}] + candidates
    return candidates + [{"type": "disagree"}]

def run_episode(q, objective, training, epsilon=0.15):
    scenario = sample_scenario()
    env = ContinualRewardEnv(
        scenario, objective=objective, max_turns=6, gap_weight=1.0
    )
    obs0, obs1 = env.reset()
    observations = [obs0, obs1]
    trajectory = []
    done = False

    while not done:
        agent = env.current_player
        obs = observations[agent]
        key = state_key(obs)
        choices = actions(obs)

        if training and random.random() < epsilon:
            action = random.choice(choices)
        else:
            scores = [q[(key, repr(a))] for a in choices]
            action = choices[max(range(len(choices)), key=lambda i: scores[i])]

        trajectory.append((key, repr(action)))
        observations[0], observations[1], rewards, done = env.step(agent, action)

    # Terminal reward is the learning signal for every visited state-action.
    if training:
        for key, action_repr in trajectory:
            q[(key, action_repr)] += 0.1 * (
                rewards[0 if key[0] == 0 else 1] - q[(key, action_repr)]
            )

    return rewards, env.outcome

def evaluate(q, objective, episodes=300):
    rewards, deals = [], 0
    for _ in range(episodes):
        result, outcome = run_episode(q, objective, training=False)
        rewards.append(sum(result) / 2)
        deals += outcome == "deal"
    return sum(rewards) / episodes, deals / episodes

def train_stage(q, objective, episodes):
    for _ in range(episodes):
        run_episode(q, objective, training=True)

def main():
    q = defaultdict(float)

    print("Initial evaluation:", evaluate(q, "self_interest"))
    train_stage(q, "self_interest", 5000)
    after_a = evaluate(q, "self_interest")
    print("After Task A — self-interest:", after_a)

    train_stage(q, "fairness", 5000)
    after_b_a = evaluate(q, "self_interest")
    after_b_b = evaluate(q, "fairness")
    print("After Task B — Task A retained:", after_b_a)
    print("After Task B — fairness:", after_b_b)
    print("Task A reward change:", round(after_b_a[0] - after_a[0], 4))
    print("Task A deal-rate change:", round(after_b_a[1] - after_a[1], 4))

if __name__ == "__main__":
    main()
