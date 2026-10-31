# Multi-Agent Communication for Continual Learning

This project investigates whether **multi-agent communication can reduce
catastrophic forgetting during sequential learning**. It includes a controlled
multi-agent scaling study, synthetic continual-learning experiments, and an
LLM LoRA prototype using reward-ranked Deal-or-No-Deal negotiation records.

> **Current finding:** the hypothesis is not established. In the fixed-compute
> scaling experiment, 2-, 4-, and 8-agent communication did not outperform
> validation-tuned fixed EWC on balanced test error. The project’s MACL signal
> is an experimental adaptive-EWC mechanism, not a verified reproduction of
> the paper or a result on real interactive LLM negotiation.

See the [full evidence-based conclusion](results/MACL_HYPOTHESIS_CONCLUSION.md)
and [multi-agent scaling results](results/MULTIAGENT_SCALING_STUDY.md).

## What this project demonstrates

- A turn-based negotiation environment with feasibility checks and explicit
  terminal rewards.
- Sequential Task A/Task B continual-learning experiments with naive, EWC,
  and partner-signal adaptive regularization baselines.
- Fixed-compute team-size comparisons for 1, 2, 4, and 8 agents.
- Validation/test seed separation, paired comparisons, bootstrap intervals,
  and self-/shuffled-communication ablations.

The controlled and scaling studies are synthetic diagnostics. The LLM pilot
uses reward-ranked historical supervised fine-tuning proxies, has substantial
Task A/B prompt overlap, and is not online MARL. Results are reported honestly
in the linked reports; they do not prove that MACL works.

## Environment setup

Use Python 3.10 or newer. Synthetic environment tests need only the Python
standard library. The synthetic continual-learning scripts require `numpy`
and `matplotlib`. The LLM experiments additionally require PyTorch, `transformers`,
`peft`, Apple MPS or a compatible accelerator, and the Qwen model from Hugging
Face.

The dependency-free environment and reward tests are:

```bash
PYTHONPATH=src python -m unittest \
  src/test_negotiation_environment.py \
  src/test_environment_contract.py \
  src/test_interactive_episodes.py
```

The small CPU-friendly synthetic smoke test can be run with:

```bash
python src/milestone1_toy.py
```

## Controlled MACL stress test

The project also includes a validation-tuned synthetic comparison with
independent validation/test seeds, equal-sized fixed-EWC and MACL
hyperparameter searches, and partner/self/shuffled communication ablations:

```bash
.venv/bin/python src/controlled_macl_study.py
```

It writes `results/CONTROLLED_MACL_STUDY.md` and the full per-seed metrics in
`results/controlled_macl_study.json`. This tests the simplified adaptive-EWC
mechanism only; it is not evidence that MACL works for LLM negotiation.

## Multi-agent scaling experiment

To compare 1, 2, 4, and 8 agents under the same team-level optimizer-update
budget, run:

```bash
.venv/bin/python src/multiagent_scaling_study.py
```

The study tunes controls on validation seeds and evaluates on separate test
seeds, including fixed EWC, local adaptive strength, peer communication, and
shuffled communication. It writes
`results/MULTIAGENT_SCALING_STUDY.md` and
`results/multiagent_scaling_study.json`. Summarize all completed studies with:

```bash
python src/analyze_macl_hypothesis.py
```

## LLM continual-learning pilot

Task A and Task B are reward-ranked historical SFT proxies, not separately
collected interactive tasks. The disjoint-prompt files remove exact prompt
overlap across tasks, but do not guarantee different scenarios or conflicting
optimal behavior. The completed matched LLM pilot used the original
overlapping-prompt splits, so it is confounded; the disjoint split has not yet
been used for a completed LLM comparison. The prototype trains two independent
LoRA agents in sequence:

1. Task A (self-interest proxy)
2. Task B (fairness proxy)
3. Task A again (a revisit, not a clean forgetting measurement)

The comparison arms are:

- `naive`: no consolidation penalty
- `ewc`: fixed-strength EWC
- `macl`: partner Fisher-weighted adapter drift controls EWC strength

The report generator summarizes the completed matched pilot (three seeds) and
the synthetic diagnostics:

```bash
python src/analyze_macl_hypothesis.py
```

It writes the per-seed analysis JSON and
`results/MACL_HYPOTHESIS_CONCLUSION.md`. Task-A retention is measured by its
validation loss change from after Task A to after Task B; lower is better.
Task-A results after the A-prime revisit are intentionally excluded from that
measure.

## Data and interpretation

To reproduce data preparation, obtain the Deal-or-No-Deal source from
[Facebook Research](https://github.com/facebookresearch/end-to-end-negotiator)
and follow its included **CC BY-NC 4.0** license. The upstream dataset and
processed training data are intentionally not committed to this portfolio
repository. Place the upstream checkout at
`data_source/end-to-end-negotiator`, then run:

```bash
python src/prepare_sft_data.py
python src/build_continual_tasks.py
python src/make_disjoint_prompt_tasks.py
```

Generated datasets, checkpoints, adapters, and model outputs are excluded from
Git to keep the repository small and avoid redistributing licensed data or
large model artifacts. The compact reports and metrics under `results/` are
included.

Do not treat a small pilot or synthetic benchmark as proof that MACL works.
A convincing claim needs more independent seeds, validation-tuned and
compute-matched baselines, genuinely distinct/conflicting tasks, and
negotiation performance and retention metrics—not only token loss.

Dataset source: <https://github.com/facebookresearch/end-to-end-negotiator>  
Paper: <https://aclanthology.org/D17-1259/>
