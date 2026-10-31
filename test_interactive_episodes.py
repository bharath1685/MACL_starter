import random
import unittest

from interactive_negotiation_env import NegotiationEnv, Scenario


def run_episode(env, policies):
    observations = env.reset()
    while not env.done:
        agent = env.current_player
        action = policies[agent](observations[agent])
        obs0, obs1, rewards, _ = env.step(agent, action)
        observations = (obs0, obs1)
    return env.outcome, env.rewards


def main():
    scenario = Scenario(stocks=(2, 1, 1), values=((3, 2, 1), (1, 1, 1)))

    def proposer(obs):
        return {"type": "offer", "allocation": [1, 0, 0]}

    def responder(obs):
        if obs["offer_from_other_agent"]:
            return {"type": "accept"}
        return {"type": "reject"}

    env = NegotiationEnv(scenario, max_turns=6, gap_weight=0.0)
    outcome, rewards = run_episode(env, [proposer, responder])
    assert outcome == "deal", f"Expected deal, got {outcome}"
    assert rewards == (3, 3), f"Unexpected rewards: {rewards}"

    print("PASS: two-agent deal episode and terminal rewards")


class InteractiveNegotiationEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(
            stocks=(2, 1, 1),
            values=((3, 2, 1), (1, 1, 1)),
        )

    def test_reset_does_not_reseed_global_random_generator(self):
        env = NegotiationEnv(self.scenario)
        random.seed(123)
        expected = random.random()
        random.seed(123)

        env.reset(seed=999)

        self.assertEqual(random.random(), expected)

    def test_max_turns_must_be_a_positive_integer(self):
        for max_turns in (0, -1, 1.5, True):
            with self.subTest(max_turns=max_turns):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    NegotiationEnv(self.scenario, max_turns=max_turns)

    def test_gap_weight_must_be_finite_and_non_negative(self):
        for gap_weight in (-1, float("nan"), float("inf"), True):
            with self.subTest(gap_weight=gap_weight):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    NegotiationEnv(self.scenario, gap_weight=gap_weight)


if __name__ == "__main__":
    main()
