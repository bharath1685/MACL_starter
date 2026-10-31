"""Minimal two-agent alternating-offer negotiation environment.

Standalone Python; no model downloads or training required.
Offers encode agent 0's allocation. Agent 1 receives the remainder.
"""
from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class Scenario:
    stocks: tuple[int, int, int]
    values: tuple[tuple[int, int, int], tuple[int, int, int]]

    def __post_init__(self):
        if len(self.stocks) != 3 or any(type(x) is not int or x < 0 for x in self.stocks):
            raise ValueError("stocks must be three non-negative integers")
        if len(self.values) != 2 or any(
            len(v) != 3 or any(type(x) is not int or x < 0 for x in v)
            for v in self.values
        ):
            raise ValueError("each agent needs three non-negative integer values")


class NegotiationEnv:
    """Turn-based environment with private values and legal stock-conserving offers."""

    def __init__(self, scenario, max_turns=10, gap_weight=1.0):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if (
            not isinstance(gap_weight, Real)
            or isinstance(gap_weight, bool)
            or not math.isfinite(gap_weight)
            or gap_weight < 0
        ):
            raise ValueError("gap_weight must be a finite non-negative number")
        self.scenario = scenario
        self.max_turns = max_turns
        self.gap_weight = gap_weight
        self.reset()

    def reset(self, seed=None):
        """Reset the deterministic environment; seed is accepted for API compatibility."""
        self.current_player = 0
        self.turns = 0
        self.offer = None
        self.offer_by = None
        self.history = []
        self.done = False
        self.outcome = None
        self.rewards = (0.0, 0.0)
        return self.observe(0), self.observe(1)

    def observe(self, agent):
        if agent not in (0, 1):
            raise ValueError("agent must be 0 or 1")
        own_offer = None
        if self.offer is not None:
            own_offer = (
                self.offer if agent == 0
                else tuple(s - x for s, x in zip(self.scenario.stocks, self.offer))
            )
        return {
            "agent": agent,
            "stocks": self.scenario.stocks,
            "private_values": self.scenario.values[agent],
            "current_player": self.current_player,
            "turns_remaining": max(0, self.max_turns - self.turns),
            "offer_exists": self.offer is not None,
            "offer_from_other_agent": self.offer is not None and self.offer_by != agent,
            "current_offer_for_self": own_offer,
            "history": tuple(dict(event) for event in self.history),
            "done": self.done,
            "outcome": self.outcome,
        }

    def payoff(self, agent, allocation):
        return sum(v * n for v, n in zip(self.scenario.values[agent], allocation))

    def _finish(self, outcome):
        self.done = True
        self.outcome = outcome
        if outcome == "deal":
            allocations = (
                self.offer,
                tuple(s - x for s, x in zip(self.scenario.stocks, self.offer)),
            )
            p0 = self.payoff(0, allocations[0])
            p1 = self.payoff(1, allocations[1])
            gap = abs(p0 - p1)
            self.rewards = (
                float(p0 - self.gap_weight * gap),
                float(p1 - self.gap_weight * gap),
            )
        else:
            self.rewards = (0.0, 0.0)
        self.history.append({"event": "terminal", "outcome": outcome})
        return self.observe(0), self.observe(1), self.rewards, True

    def step(self, agent, action):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if agent != self.current_player:
            raise ValueError(f"agent {self.current_player} must act next")
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            raise ValueError("action must be a dict with a string 'type'")

        kind = action["type"]
        if kind == "offer":
            raw = action.get("allocation")
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                raise ValueError("offer allocation must contain three integers")
            if any(type(x) is not int for x in raw):
                raise ValueError("offer quantities must be integers")
            allocation = tuple(raw)
            if any(x < 0 or x > s for x, s in zip(allocation, self.scenario.stocks)):
                raise ValueError("offer violates stock constraints")
            self.offer = allocation
            self.offer_by = agent
            self.history.append({"event": "offer", "agent": agent, "agent0_allocation": allocation})
        elif kind == "accept":
            if self.offer is None or self.offer_by == agent:
                raise ValueError("only an outstanding offer from the other agent can be accepted")
            self.history.append({"event": "accept", "agent": agent})
            self.turns += 1
            return self._finish("deal")
        elif kind == "reject":
            if self.offer is None or self.offer_by == agent:
                raise ValueError("only an outstanding offer from the other agent can be rejected")
            self.history.append({"event": "reject", "agent": agent})
            self.offer = None
            self.offer_by = None
        elif kind == "disagree":
            self.history.append({"event": "disagree", "agent": agent})
            self.turns += 1
            return self._finish("disagree")
        elif kind == "disconnect":
            self.history.append({"event": "disconnect", "agent": agent})
            self.turns += 1
            return self._finish("disconnect")
        else:
            raise ValueError(f"unknown action type: {kind}")

        self.turns += 1
        if self.turns >= self.max_turns:
            return self._finish("no_agreement")
        self.current_player = 1 - agent
        return self.observe(0), self.observe(1), (0.0, 0.0), False


def self_test():
    scenario = Scenario(
        stocks=(2, 1, 1),
        values=((3, 2, 1), (1, 1, 1)),
    )
    env = NegotiationEnv(scenario, max_turns=4, gap_weight=0.0)
    env.step(0, {"type": "offer", "allocation": [1, 0, 0]})
    obs0, obs1, rewards, done = env.step(1, {"type": "accept"})
    assert done and obs0["outcome"] == "deal"
    assert rewards == (3.0, 3.0), rewards
    assert obs0["current_offer_for_self"] == (1, 0, 0)
    assert obs1["current_offer_for_self"] == (1, 1, 1)

    env.reset()
    try:
        env.step(0, {"type": "offer", "allocation": [3, 0, 0]})
        raise AssertionError("illegal offer was accepted")
    except ValueError:
        pass

    env.reset()
    env.step(0, {"type": "offer", "allocation": [1, 0, 0]})
    env.step(1, {"type": "reject"})
    assert env.current_player == 0 and env.offer is None

    env.reset()
    env.step(0, {"type": "disagree"})
    assert env.done and env.outcome == "disagree" and env.rewards == (0.0, 0.0)
    print("PASS: deal/payoffs, private observations, stock validation, rejection, disagreement")


if __name__ == "__main__":
    self_test()
