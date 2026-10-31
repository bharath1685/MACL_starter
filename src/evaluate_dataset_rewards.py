
from pathlib import Path
from collections import Counter
import re
import statistics

DATA_FILE = Path(
    "data_source/end-to-end-negotiator/src/data/negotiate/train.txt"
)

GAP_WEIGHT = 1.0  # Illustrative setting, not a paper-verified parameter.

def extract(line, tag):
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", line)
    if not match:
        raise ValueError(f"Missing <{tag}> field")
    return match.group(1).strip()

def repeated_tag(output, tag):
    return bool(re.fullmatch(rf"(<{tag}>\s*)+", output))

def payoff(values, allocation):
    return sum(value * quantity for value, quantity in zip(values, allocation))

def main():
    outcomes = Counter()
    own_payoffs = []
    partner_payoffs = []
    fairness_own_rewards = []
    fairness_partner_rewards = []
    errors = []

    with DATA_FILE.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                own_input = tuple(
                    int(x) for x in extract(line, "input").split()
                )
                partner_input = tuple(
                    int(x) for x in extract(line, "partner_input").split()
                )
                output = extract(line, "output")

                counts = own_input[0::2]
                own_values = own_input[1::2]
                partner_counts = partner_input[0::2]
                partner_values = partner_input[1::2]

                if len(own_input) != 6 or len(partner_input) != 6:
                    raise ValueError("Expected six input integers")

                if counts != partner_counts:
                    raise ValueError("Item counts do not match")

                if repeated_tag(output, "disagree"):
                    outcomes["disagree"] += 1
                    continue
                if repeated_tag(output, "no_agreement"):
                    outcomes["no_agreement"] += 1
                    continue
                if repeated_tag(output, "disconnect"):
                    outcomes["disconnect"] += 1
                    continue

                pairs = re.findall(r"(item[0-2])=(\d+)", output)
                labels = [label for label, value in pairs]
                expected = [
                    "item0", "item1", "item2",
                    "item0", "item1", "item2"
                ]

                if labels != expected:
                    raise ValueError("Unexpected allocation format")

                allocation = [int(value) for label, value in pairs]
                own_allocation = allocation[:3]
                partner_allocation = allocation[3:]

                if any(
                    own_allocation[i] + partner_allocation[i] != counts[i]
                    for i in range(3)
                ):
                    raise ValueError("Allocation does not match item counts")

                own_pay = payoff(own_values, own_allocation)
                partner_pay = payoff(partner_values, partner_allocation)
                gap = abs(own_pay - partner_pay)

                own_payoffs.append(own_pay)
                partner_payoffs.append(partner_pay)
                fairness_own_rewards.append(own_pay - GAP_WEIGHT * gap)
                fairness_partner_rewards.append(
                    partner_pay - GAP_WEIGHT * gap
                )
                outcomes["valid deal allocation"] += 1

            except Exception as error:
                errors.append((line_number, str(error)))

    print(f"Records with recognized outcomes: {sum(outcomes.values())}")
    print(f"Records needing investigation: {len(errors)}")
    print("\nOutcome counts:")
    for name, count in sorted(outcomes.items()):
        print(f"  {name}: {count}")

    if own_payoffs:
        print("\nSuccessful-allocation payoff summary:")
        print(f"  Deal records: {len(own_payoffs)}")
        print(f"  Mean own payoff: {statistics.mean(own_payoffs):.3f}")
        print(f"  Mean partner payoff: {statistics.mean(partner_payoffs):.3f}")
        print(
            "  Mean illustrative own fairness reward: "
            f"{statistics.mean(fairness_own_rewards):.3f}"
        )
        print(
            "  Mean illustrative partner fairness reward: "
            f"{statistics.mean(fairness_partner_rewards):.3f}"
        )
        print(f"  Illustrative gap weight: {GAP_WEIGHT}")

    for line_number, error in errors[:10]:
        print(f"Line {line_number}: {error}")

if __name__ == "__main__":
    main()
