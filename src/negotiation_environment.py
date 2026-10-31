"""Small, tested negotiation environment primitives for the MACL project.

This module does not implement an RL algorithm or an interactive turn protocol.
It evaluates completed negotiation outcomes using the project's reward_engine.
Terminal rewards for non-agreement and disconnect MUST be supplied explicitly.
"""
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple
import re

from reward_engine import (
    Scenario,
    parse_input,
    validate_allocation,
    payoff,
    self_interested_rewards,
    fairness_shaped_rewards,
)

_TASKS = {
    "task1_self_interested",
    "task2_fairness",
    "task3_self_interested",
}
_TAG_RE = re.compile(r"<(input|partner_input|output)>(.*?)</\1>", re.I | re.S)
_ASSIGNMENT_RE = re.compile(r"item([0-2])\s*=\s*(\d+)", re.I)


@dataclass(frozen=True)
class OutcomeEvaluation:
    outcome: str
    task: str
    own_payoff: Optional[float]
    partner_payoff: Optional[float]
    own_reward: float
    partner_reward: float
    allocation_own: Optional[Tuple[int, int, int]] = None
    allocation_partner: Optional[Tuple[int, int, int]] = None


def classify_terminal_output(output: str) -> str:
    """Return 'deal_candidate', 'no_agreement', or 'disconnect'."""
    text = (output or "").strip().lower()
    if not text:
        raise ValueError("Output is empty.")
    if "disconnect" in text:
        return "disconnect"
    if any(token in text for token in ("no_agreement", "no_deal", "no deal", "disagree")):
        return "no_agreement"
    if len(_ASSIGNMENT_RE.findall(text)) == 6:
        return "deal_candidate"
    raise ValueError(f"Unrecognized negotiation output: {output!r}")


def parse_allocation(output: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Parse six item assignments in order: own item0..2, partner item0..2."""
    matches = _ASSIGNMENT_RE.findall(output or "")
    labels = [int(item_index) for item_index, _ in matches]
    if labels != [0, 1, 2, 0, 1, 2]:
        raise ValueError(
            "Expected assignments in order item0, item1, item2 for each agent."
        )
    values = [int(value) for _, value in matches]
    return tuple(values[:3]), tuple(values[3:])


def task_rewards(
    task: str,
    own_payoff: float,
    partner_payoff: float,
    *,
    gap_weight: float = 1.0,
) -> Tuple[float, float]:
    """Compute deal rewards. gap_weight is a project setting, not paper-verified."""
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(_TASKS)}")
    if task in ("task1_self_interested", "task3_self_interested"):
        return self_interested_rewards(own_payoff, partner_payoff)
    return fairness_shaped_rewards(own_payoff, partner_payoff, gap_weight=gap_weight)


def evaluate_completed_outcome(
    input_text: str,
    partner_input_text: str,
    output: str,
    task: str,
    *,
    terminal_rewards: Optional[Mapping[str, Sequence[float]]] = None,
    gap_weight: float = 1.0,
) -> OutcomeEvaluation:
    """Evaluate a completed record.

    terminal_rewards must map 'no_agreement' and/or 'disconnect' to a pair
    (own_reward, partner_reward). Missing terminal policy raises an error rather
    than silently inventing a penalty. Deal payoffs are calculated from the
    recorded allocation and private values.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(_TASKS)}")
    scenario: Scenario = parse_input(input_text, partner_input_text)
    outcome = classify_terminal_output(output)

    if outcome != "deal_candidate":
        if terminal_rewards is None or outcome not in terminal_rewards:
            raise ValueError(
                f"No explicit terminal reward configured for {outcome!r}. "
                "Pass terminal_rewards={outcome: (own_reward, partner_reward)}."
            )
        pair = terminal_rewards[outcome]
        if len(pair) != 2:
            raise ValueError(f"Terminal reward for {outcome!r} must contain two values.")
        own_reward, partner_reward = float(pair[0]), float(pair[1])
        return OutcomeEvaluation(
            outcome=outcome, task=task, own_payoff=None, partner_payoff=None,
            own_reward=own_reward, partner_reward=partner_reward,
        )

    own_allocation, partner_allocation = parse_allocation(output)
    if not validate_allocation(scenario, own_allocation, partner_allocation):
        raise ValueError("Recorded allocation is infeasible for this scenario.")
    own_pay = payoff(scenario.own_values, own_allocation)
    partner_pay = payoff(scenario.partner_values, partner_allocation)
    own_reward, partner_reward = task_rewards(
        task, own_pay, partner_pay, gap_weight=gap_weight
    )
    return OutcomeEvaluation(
        outcome="deal", task=task,
        own_payoff=float(own_pay), partner_payoff=float(partner_pay),
        own_reward=float(own_reward), partner_reward=float(partner_reward),
        allocation_own=own_allocation, allocation_partner=partner_allocation,
    )


def parse_dataset_line(line: str) -> dict:
    """Parse required tagged fields from one original dataset line."""
    fields = {m.group(1).lower(): m.group(2).strip() for m in _TAG_RE.finditer(line)}
    required = {"input", "partner_input", "output"}
    missing = required - fields.keys()
    if missing:
        raise ValueError(f"Dataset record missing tags: {sorted(missing)}")
    return fields
