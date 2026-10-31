"""Unit tests for negotiation_environment.py.

Run from project root:
PYTHONPATH=src python -m unittest src/test_negotiation_environment.py
"""
import unittest

from negotiation_environment import (
    classify_terminal_output,
    parse_allocation,
    task_rewards,
    evaluate_completed_outcome,
    parse_dataset_line,
)

OWN_INPUT = "1 4 4 1 1 2"
PARTNER_INPUT = "1 0 4 2 1 2"
DEAL = "item0=1 item1=0 item2=1 item0=0 item1=4 item2=0"


class NegotiationEnvironmentTests(unittest.TestCase):
    def test_classifies_known_outcomes(self):
        self.assertEqual(classify_terminal_output(DEAL), "deal_candidate")
        self.assertEqual(classify_terminal_output("<disagree> " * 6), "no_agreement")
        self.assertEqual(classify_terminal_output("<no_agreement> " * 6), "no_agreement")
        self.assertEqual(classify_terminal_output("<disconnect> " * 6), "disconnect")

    def test_rejects_unknown_output(self):
        with self.assertRaises(ValueError):
            classify_terminal_output("some unexpected text")

    def test_parses_allocation_in_recorded_order(self):
        self.assertEqual(parse_allocation(DEAL), ((1, 0, 1), (0, 4, 0)))

    def test_self_interested_task_rewards(self):
        self.assertEqual(task_rewards("task1_self_interested", 6, 4), (6.0, 4.0))
        self.assertEqual(task_rewards("task3_self_interested", 6, 4), (6.0, 4.0))

    def test_fairness_task_rewards(self):
        self.assertEqual(task_rewards("task2_fairness", 6, 4, gap_weight=1.0), (4.0, 2.0))

    def test_fairness_reward_rejects_non_finite_or_invalid_weights(self):
        for gap_weight in (-1, float("nan"), float("inf"), True):
            with self.subTest(gap_weight=gap_weight):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    task_rewards("task2_fairness", 6, 4, gap_weight=gap_weight)

    def test_evaluates_recorded_deal(self):
        result = evaluate_completed_outcome(
            OWN_INPUT, PARTNER_INPUT, DEAL, "task1_self_interested"
        )
        self.assertEqual(result.outcome, "deal")
        self.assertEqual(result.allocation_own, (1, 0, 1))
        self.assertEqual(result.allocation_partner, (0, 4, 0))
        self.assertEqual(result.own_payoff, 6.0)
        self.assertEqual(result.partner_payoff, 8.0)
        self.assertEqual((result.own_reward, result.partner_reward), (6.0, 8.0))

    def test_infeasible_deal_fails(self):
        invalid = "item0=2 item1=0 item2=1 item0=0 item1=4 item2=0"
        with self.assertRaises(ValueError):
            evaluate_completed_outcome(
                OWN_INPUT, PARTNER_INPUT, invalid, "task1_self_interested"
            )

    def test_terminal_reward_must_be_explicit(self):
        with self.assertRaisesRegex(ValueError, "No explicit terminal reward"):
            evaluate_completed_outcome(
                OWN_INPUT, PARTNER_INPUT, "<disconnect> " * 6,
                "task1_self_interested"
            )

    def test_explicit_terminal_reward_is_used(self):
        result = evaluate_completed_outcome(
            OWN_INPUT, PARTNER_INPUT, "<disconnect> " * 6,
            "task2_fairness",
            terminal_rewards={"disconnect": (0.0, 0.0)},
        )
        self.assertEqual(result.outcome, "disconnect")
        self.assertIsNone(result.own_payoff)
        self.assertEqual((result.own_reward, result.partner_reward), (0.0, 0.0))

    def test_parses_dataset_record_tags(self):
        line = (
            "<input>1 4 4 1 1 2</input>"
            "<partner_input>1 0 4 2 1 2</partner_input>"
            "<output>item0=1 item1=0 item2=1 item0=0 item1=4 item2=0</output>"
        )
        fields = parse_dataset_line(line)
        self.assertEqual(fields["input"], OWN_INPUT)
        self.assertEqual(fields["partner_input"], PARTNER_INPUT)

    def test_unknown_task_fails(self):
        with self.assertRaises(ValueError):
            task_rewards("task4", 1, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
