"""Validated allocation and reward utilities.

The fairness-shaped reward is an illustrative research choice, not a
verified transcription of the MACL paper's objective.
"""
from dataclasses import dataclass
import math
from numbers import Integral
from numbers import Real
from typing import Sequence


@dataclass(frozen=True)
class Scenario:
    counts: tuple[int, int, int]
    own_values: tuple[int, int, int]
    partner_values: tuple[int, int, int]


def _six_nonnegative_integers(text: str, field_name: str) -> list[int]:
    if not isinstance(text, str):
        raise TypeError(f"{field_name} must be a string of six integers.")

    parts = text.split()
    if len(parts) != 6:
        raise ValueError(f"{field_name} must contain exactly six integers.")

    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain integers only.") from exc

    if any(value < 0 for value in values):
        raise ValueError(f"{field_name} cannot contain negative values.")

    return values


def parse_input(input_text: str, partner_input_text: str) -> Scenario:
    own = _six_nonnegative_integers(input_text, "input_text")
    partner = _six_nonnegative_integers(
        partner_input_text, "partner_input_text"
    )

    # Dataset format: count0, value0, count1, value1, count2, value2.
    counts = tuple(own[0::2])
    partner_counts = tuple(partner[0::2])

    if counts != partner_counts:
        raise ValueError("Both agents must have matching item counts.")

    return Scenario(
        counts=counts,
        own_values=tuple(own[1::2]),
        partner_values=tuple(partner[1::2]),
    )


def validate_allocation(
    scenario: Scenario,
    own: Sequence[int],
    partner: Sequence[int],
) -> bool:
    if len(own) != 3 or len(partner) != 3:
        return False

    for a, b, count in zip(own, partner, scenario.counts):
        if not isinstance(a, Integral) or isinstance(a, bool):
            return False
        if not isinstance(b, Integral) or isinstance(b, bool):
            return False
        if a < 0 or b < 0 or a + b != count:
            return False

    return True


def payoff(values: Sequence[int], allocation: Sequence[int]) -> int:
    if len(values) != 3 or len(allocation) != 3:
        raise ValueError("Expected exactly three values and quantities.")

    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0
           for v in values):
        raise ValueError("Values must be non-negative integers.")

    if any(not isinstance(q, Integral) or isinstance(q, bool) or q < 0
           for q in allocation):
        raise ValueError("Allocation quantities must be non-negative integers.")

    return sum(v * q for v, q in zip(values, allocation))


def self_interested_rewards(
    own_payoff: float, partner_payoff: float
) -> tuple[float, float]:
    """Independent-payoff baseline: each agent values its own payoff."""
    return float(own_payoff), float(partner_payoff)


def fairness_shaped_rewards(
    own_payoff: float,
    partner_payoff: float,
    gap_weight: float = 1.0,
) -> tuple[float, float]:
    """Illustrative gap-penalty reward; verify against the paper before use."""
    if (
        not isinstance(gap_weight, Real)
        or isinstance(gap_weight, bool)
        or not math.isfinite(gap_weight)
        or gap_weight < 0
    ):
        raise ValueError("gap_weight must be a finite non-negative number.")

    gap = abs(own_payoff - partner_payoff)
    return (
        float(own_payoff - gap_weight * gap),
        float(partner_payoff - gap_weight * gap),
    )


def run_tests() -> None:
    scenario = parse_input("1 4 4 1 1 2", "1 4 4 1 1 2")

    own_allocation = (1, 0, 1)
    partner_allocation = (0, 4, 0)

    assert scenario.counts == (1, 4, 1)
    assert validate_allocation(scenario, own_allocation, partner_allocation)
    assert not validate_allocation(scenario, (2, 0, 1), partner_allocation)
    assert not validate_allocation(scenario, (1, -1, 1), partner_allocation)

    own_pay = payoff(scenario.own_values, own_allocation)
    partner_pay = payoff(scenario.partner_values, partner_allocation)
    assert own_pay == 6
    assert partner_pay == 4

    assert self_interested_rewards(own_pay, partner_pay) == (6.0, 4.0)
    assert fairness_shaped_rewards(6, 4, 1.0) == (4.0, 2.0)
    assert fairness_shaped_rewards(5, 5, 1.0) == (5.0, 5.0)

    for bad in ("1 2 3", "1 2 x 4 5 6", "-1 2 3 4 5 6"):
        try:
            parse_input(bad, "1 2 3 4 5 6")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid input was accepted: {bad!r}")

    try:
        fairness_shaped_rewards(1, 2, -0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative gap_weight was accepted.")

    print("PASS: reward engine validation tests")
    print("Example payoffs:", own_pay, partner_pay)
    print("Illustrative fairness rewards:",
          fairness_shaped_rewards(own_pay, partner_pay))


if __name__ == "__main__":
    run_tests()
