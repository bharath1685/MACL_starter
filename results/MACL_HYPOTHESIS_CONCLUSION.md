# MACL and catastrophic forgetting: evidence-based conclusion

## Conclusion

The current evidence does **not** establish that multi-agent communication mitigates catastrophic forgetting better than fixed EWC. The controlled equal-search-budget synthetic study below found lower Task-A forgetting with MACL, but worse Task-B learning and worse overall balanced error than validation-tuned fixed EWC. The LLM results are a small, overlapping-task pilot, not a confirmation or a reproduction of the MACL paper.

## Matched LLM pilot (confounded)

The three seeds use matched initialization and training budgets, but the runs use the original Task A/B data, which have 1,528 exact prompts in common (70.1% of Task A's unique prompts). Thus this is a confounded pilot, not evidence from independent tasks. Lower validation NLL is better. Retention change is Task-A NLL after Task B minus Task-A NLL after Task A; negative values mean Task A improved rather than being forgotten. Values are mean +/- sample standard deviation over seeds.

| Method | Task-A NLL after A | Task-A NLL after B | Task-A change after B | Task-B NLL after B |
|---|---:|---:|---:|---:|
| naive | 0.02590 +/- 0.00911 | 0.02634 +/- 0.01288 | 0.00044 +/- 0.01527 | 0.02758 +/- 0.00630 |
| ewc | 0.02590 +/- 0.00911 | 0.02372 +/- 0.00756 | -0.00218 +/- 0.00410 | 0.02657 +/- 0.01510 |
| macl | 0.02590 +/- 0.00911 | 0.02318 +/- 0.00664 | -0.00271 +/- 0.00660 | 0.02555 +/- 0.01143 |

Paired MACL-minus-EWC Task-A change: -0.00053 +/- 0.00261; MACL has a lower (better) change on 1/3 seeds. This pilot is confounded by task overlap and only three seeds, so it cannot establish a reliable effect.

## Five-seed synthetic diagnostic

Task-A score change after learning Task B (negative MSE; higher is better):

| Method | Mean change +/- sample SD |
|---|---:|
| naive | -17.84993 +/- 0.97098 |
| fixed_ewc | -17.17706 +/- 0.93729 |
| macl_toy | -17.17398 +/- 0.93680 |

MACL-minus-fixed-EWC paired score change: 0.00308 +/- 0.00205; MACL is higher on 5/5 seeds. Both methods modestly outperform naive learning on this deliberately conflicting synthetic task, but substantial forgetting remains and their difference is negligible.

The mean retention change is -17.84993 for naive, -17.17706 for fixed EWC, and -17.17398 for MACL. Thus consolidation improves the score by about 0.67 over naive, while the MACL increment over fixed EWC is only 0.00308 (about 0.02% of the EWC forgetting magnitude).


## Controlled synthetic study: 100 held-out seeds

This study tunes fixed EWC and the adaptive MACL mapping on 32 separate validation seeds, with an equal 108-candidate hyperparameter search for each. It then evaluates on 100 unseen seeds. Lower errors are better.

| Paired difference (MACL minus comparator) | Mean | Bootstrap 95% CI |
|---|---:|---:|
| Balanced test MSE vs tuned fixed EWC | 0.03625 | [0.02474, 0.04897] |
| Task-A forgetting vs tuned fixed EWC | -0.27780 | [-0.40467, -0.15032] |
| Task-B test MSE vs tuned fixed EWC | 0.35029 | [0.22553, 0.47826] |
| Balanced test MSE vs self-signal ablation | 0.00609 | [-0.00235, 0.01671] |

Interpretation: MACL retained Task A better than tuned EWC on average, but Task-B error increased enough that its balanced test MSE was significantly worse. Partner communication did not outperform the self-signal ablation (its balanced-error interval includes zero). Therefore this experiment does **not** show MACL outperforms fixed EWC, and does not isolate a reliable benefit from multi-agent communication.

Full protocol, ablations, and per-seed results: [CONTROLLED_MACL_STUDY.md](./CONTROLLED_MACL_STUDY.md) and [controlled_macl_study.json](./controlled_macl_study.json).


## Agent-count scaling: 100 held-out seeds per team size

Every team size received 320 optimizer updates per task. EWC and adaptive methods each used 108 validation candidates over 24 separate seeds. Values below are paired peer-communication minus fixed-EWC balanced test MSE; positive values favor EWC.

| Agents | Difference | Bootstrap 95% CI | Peer wins |
|---:|---:|---:|---:|
| 1 | 0.02913 | [0.01682, 0.04261] | 34/100 |
| 2 | 0.09739 | [0.07627, 0.11984] | 14/100 |
| 4 | 0.02509 | [0.01719, 0.03448] | 23/100 |
| 8 | 0.01461 | [0.01078, 0.01877] | 23/100 |

No tested multi-agent team size outperformed tuned fixed EWC on balanced test error; all paired intervals favor fixed EWC. Eight agents did outperform the local adaptive no-communication variant on balanced error, but not the shuffled-message ablation, and its Task-B error was higher. This does not show that communication itself caused the gain or that adding agents improves performance.

Full scaling report: [MULTIAGENT_SCALING_STUDY.md](./MULTIAGENT_SCALING_STUDY.md) and [multiagent_scaling_study.json](./multiagent_scaling_study.json).
## Limitations and interpretation

- The LLM task labels are reward-ranked historical SFT proxies, not interactive negotiation tasks or online MARL.
- A cleaned exact-prompt-disjoint dataset exists, but these completed LLM pilots did not use it; even that split would not guarantee distinct scenarios or genuinely conflicting objectives.
- The MACL arm exchanges a scalar adapter-drift signal to set an EWC coefficient; this is a project prototype, not a verified implementation of the paper's algorithm.
- Three LLM seeds and short runs are insufficient for a confirmatory claim, and the reported LLM pilot has substantial exact-prompt overlap.
- Task A-prime performance is not used as a forgetting measure because Task A is trained again during that stage.

## Verdict

**Hypothesis status: not demonstrated by the current project evidence.** Current results support that EWC-style consolidation can reduce forgetting in a toy setting; they do not show a meaningful, reliable benefit from adding communication over a matched fixed-EWC baseline. A larger pre-registered, multi-seed study on genuinely distinct tasks, with tuned fixed-EWC controls and negotiation reward/retention metrics, is required before claiming that MACL works.

Machine-readable metrics: [macl_hypothesis_analysis.json](./macl_hypothesis_analysis.json).
