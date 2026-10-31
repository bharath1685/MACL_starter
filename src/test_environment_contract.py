"""Contract tests for MACL task objectives and negotiation outcomes.

These tests document project decisions; they do not claim to reproduce
an unspecified numerical reward formula from the MACL paper.
"""
import unittest

from reward_engine import (
    fairness_shaped_rewards,
    self_interested_rewards,
    parse_input,
    validate_allocation,
    payoff,
)


def task_rewards(task, own_payoff, partner_payoff, gap_weight=1.0):
    if task in ("task1_self_interested", "task3_self_interested"):
        return self_interested_rewards(own_payoff, partner_payoff)
    if task == "task2_fairness":
        return fairness_shaped_rewards(
            own_payoff, partner_payoff, gap_weight=gap_weight
        )
    raise ValueError(f"Unknown task: {task}")


def terminal_outcome(output):
    """Classify known dataset terminal outcomes; do not invent payoffs."""
    text = (output or "").lower()
    if "disconnect" in text:
        return "disconnect"
    if any(token in text for token in (
        "disagree", "no_agreement", "no_deal", "no deal"
    )):
        return "no_agreement"
    return "allocation_candidate"


class EnvironmentContractTests(unittest.TestCase):
    def test_task1_and_task3_have_same_objective(self):
        for own, partner in ((8, 3), (5, 5), (0, 7)):
            self.assertEqual(
                task_rewards("task1_self_interested", own, partner),
                task_rewards("task3_self_interested", own, partner),
            )

    def test_self_interested_rewards_preserve_individual_payoffs(self):
        self.assertEqual(task_rewards("task1_self_interested", 8, 3), (8.0, 3.0))

    def test_fairness_reward_penalizes_gap_symmetrically(self):
        rewards = task_rewards("task2_fairness", 8, 3, gap_weight=1.0)
        self.assertEqual(rewards, (3.0, -2.0))

    def test_fairness_reward_has_no_gap_penalty_when_equal(self):
        self.assertEqual(
            task_rewards("task2_fairness", 5, 5, gap_weight=1.0),
            (5.0, 5.0),
        )

    def test_unknown_task_fails_loudly(self):
        with self.assertRaises(ValueError):
            task_rewards("unknown", 1, 1)

    def test_known_terminal_outcomes_are_distinguished(self):
        self.assertEqual(terminal_outcome("<disconnect>"), "disconnect")
        self.assertEqual(terminal_outcome("<no_agreement>"), "no_agreement")
        self.assertEqual(terminal_outcome("<disagree>"), "no_agreement")
        self.assertEqual(
            terminal_outcome("item0=1 item1=0 item2=1 item0=0 item1=1 item2=0"),
            "allocation_candidate",
        )

    def test_allocation_feasibility_and_payoffs(self):
        scenario = parse_input("1 4 4 1 1 2", "1 4 4 1 1 2")
        own = (1, 0, 1)
        partner = (0, 4, 0)
        self.assertTrue(validate_allocation(scenario, own, partner))
        self.assertEqual(payoff(scenario.own_values, own), 6)
        self.assertEqual(payoff(scenario.partner_values, partner), 4)
        self.assertFalse(validate_allocation(scenario, (2, 0, 1), partner))


if __name__ == "__main__":
    unittest.main(verbosity=2)
