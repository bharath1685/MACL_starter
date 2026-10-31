import unittest

from multiagent_scaling_study import (
    TEAM_SIZES,
    TOTAL_UPDATES_PER_TASK,
    _candidate_settings,
    _received_signals,
    _steps_per_agent,
    make_team,
)


class MultiagentScalingStudyTests(unittest.TestCase):
    def test_compute_budget_is_constant_across_team_sizes(self):
        for team_size in TEAM_SIZES:
            with self.subTest(team_size=team_size):
                self.assertEqual(
                    team_size * _steps_per_agent(team_size),
                    TOTAL_UPDATES_PER_TASK,
                )

    def test_hyperparameter_search_sizes_are_matched(self):
        self.assertEqual(
            len(_candidate_settings("fixed_ewc")),
            len(_candidate_settings("peer_communication")),
        )
        self.assertEqual(
            len(_candidate_settings("local_adaptive")),
            len(_candidate_settings("shuffled_communication")),
        )

    def test_peer_signal_excludes_the_receiving_agent(self):
        team = {
            "agents": [
                {"signal": 1.0},
                {"signal": 4.0},
                {"signal": 10.0},
            ]
        }
        self.assertEqual(
            _received_signals(team, "peer_communication"),
            [7.0, 5.5, 2.5],
        )

    def test_single_agent_communication_falls_back_to_local_signal(self):
        team = {"agents": [{"signal": 3.25}]}
        self.assertEqual(
            _received_signals(team, "peer_communication"),
            [3.25],
        )

    def test_team_generation_is_reproducible_and_has_requested_size(self):
        first = make_team(123, 4)
        second = make_team(123, 4)
        self.assertEqual(len(first["agents"]), 4)
        self.assertEqual(
            [agent["signal"] for agent in first["agents"]],
            [agent["signal"] for agent in second["agents"]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
