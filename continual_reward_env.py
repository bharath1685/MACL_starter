from src.interactive_negotiation_env import Scenario, NegotiationEnv


class ContinualRewardEnv(NegotiationEnv):
    """Negotiation environment with an explicit self-interest/fairness objective."""

    def __init__(
        self,
        scenario,
        objective="self_interest",
        max_turns=10,
        gap_weight=1.0,
    ):
        if objective not in ("self_interest", "fairness"):
            raise ValueError("objective must be 'self_interest' or 'fairness'")
        self.objective = objective
        super().__init__(
            scenario,
            max_turns=max_turns,
            gap_weight=gap_weight,
        )

    def _finish(self, outcome):
        result = super()._finish(outcome)

        if outcome == "deal" and self.objective == "self_interest":
            allocations = (
                self.offer,
                tuple(
                    s - x
                    for s, x in zip(self.scenario.stocks, self.offer)
                ),
            )
            self.rewards = (
                float(self.payoff(0, allocations[0])),
                float(self.payoff(1, allocations[1])),
            )
            result = (
                self.observe(0),
                self.observe(1),
                self.rewards,
                True,
            )

        return result


def self_test():
    scenario = Scenario(
        stocks=(2, 1, 1),
        values=((3, 2, 1), (1, 1, 1)),
    )

    env_a = ContinualRewardEnv(
        scenario, objective="self_interest", gap_weight=1.0
    )
    env_a.step(0, {"type": "offer", "allocation": [1, 0, 0]})
    _, _, rewards_a, done_a = env_a.step(1, {"type": "accept"})
    assert done_a and rewards_a == (3.0, 3.0), rewards_a

    env_b = ContinualRewardEnv(
        scenario, objective="fairness", gap_weight=1.0
    )
    env_b.step(0, {"type": "offer", "allocation": [1, 0, 0]})
    _, _, rewards_b, done_b = env_b.step(1, {"type": "accept"})
    assert done_b and rewards_b == (3.0, 3.0), rewards_b

    scenario_gap = Scenario(
        stocks=(2, 0, 0),
        values=((5, 0, 0), (1, 0, 0)),
    )
    env_a = ContinualRewardEnv(scenario_gap, objective="self_interest")
    env_a.step(0, {"type": "offer", "allocation": [2, 0, 0]})
    _, _, rewards_a, _ = env_a.step(1, {"type": "accept"})

    env_b = ContinualRewardEnv(scenario_gap, objective="fairness")
    env_b.step(0, {"type": "offer", "allocation": [2, 0, 0]})
    _, _, rewards_b, _ = env_b.step(1, {"type": "accept"})

    assert rewards_a == (10.0, 0.0), rewards_a
    assert rewards_b == (0.0, -10.0), rewards_b
    print("PASS: objective switch produces distinct self-interest and fairness rewards")


if __name__ == "__main__":
    self_test()
