# Multi-agent communication scaling study

## Protocol

This synthetic linear-regression study tests whether exchanging estimated Task-A interference across a team improves retention. Team sizes are 1, 2, 4, and 8. Each team receives exactly 320 optimizer updates per task; validation and test sets use 24 and 100 independent seeds. Fixed EWC and each adaptive arm receive equal 108-candidate validation searches. All methods share the same generated tasks and test seeds at each team size.

All scores are held-out means over teams; lower is better. Task-A retention change is Task-A test MSE after Task B minus after Task A. The 95% intervals are bootstrap intervals over test seeds.

## Results by team size

| Agents | Method | Balanced MSE | Task-A retention change | Task-B MSE |
|---:|---|---:|---:|---:|
| 1 | naive | 7.8948 [7.0335, 8.7566] | 15.7759 [14.0533, 17.4997] | 0.0068 |
| 1 | fixed_ewc | 4.0541 [3.6166, 4.4900] | 3.9106 [3.4648, 4.3628] | 4.1906 |
| 1 | local_adaptive | 4.0832 [3.6430, 4.5224] | 3.5582 [3.2351, 3.8824] | 4.6013 |
| 1 | peer_communication | 4.0832 [3.6430, 4.5224] | 3.5582 [3.2351, 3.8824] | 4.6013 |
| 1 | shuffled_communication | 4.1317 [3.6849, 4.5799] | 3.8214 [3.3125, 4.3636] | 4.4351 |
| 2 | naive | 7.8262 [7.2272, 8.4076] | 15.6387 [14.4407, 16.8016] | 0.0068 |
| 2 | fixed_ewc | 4.0045 [3.7006, 4.2967] | 3.9191 [3.6147, 4.2196] | 4.0830 |
| 2 | local_adaptive | 4.0341 [3.7340, 4.3229] | 4.4539 [4.1711, 4.7304] | 3.6075 |
| 2 | peer_communication | 4.1019 [3.7936, 4.3996] | 3.2805 [3.0584, 3.5093] | 4.9164 |
| 2 | shuffled_communication | 4.0784 [3.7688, 4.3771] | 3.8747 [3.5211, 4.2310] | 4.2753 |
| 4 | naive | 7.7516 [7.3014, 8.1943] | 15.4861 [14.5856, 16.3711] | 0.0092 |
| 4 | fixed_ewc | 4.0334 [3.7970, 4.2649] | 3.9234 [3.6904, 4.1547] | 4.1354 |
| 4 | local_adaptive | 4.0591 [3.8279, 4.2861] | 4.4318 [4.2269, 4.6326] | 3.6785 |
| 4 | peer_communication | 4.0585 [3.8214, 4.2925] | 3.7307 [3.5781, 3.8785] | 4.3784 |
| 4 | shuffled_communication | 4.0579 [3.8181, 4.2928] | 3.7462 [3.4925, 4.0092] | 4.3616 |
| 8 | naive | 6.9294 [6.6434, 7.2088] | 13.6747 [13.1070, 14.2292] | 0.1167 |
| 8 | fixed_ewc | 4.0641 [3.8951, 4.2300] | 3.9005 [3.7303, 4.0673] | 4.1603 |
| 8 | local_adaptive | 4.0951 [3.9293, 4.2593] | 4.3594 [4.2098, 4.5091] | 3.7634 |
| 8 | peer_communication | 4.0787 [3.9076, 4.2479] | 3.6066 [3.4930, 3.7205] | 4.4834 |
| 8 | shuffled_communication | 4.0800 [3.9096, 4.2466] | 3.6186 [3.4483, 3.7868] | 4.4740 |

## Paired peer-communication comparisons

A negative peer-minus-control value favors peer communication:

| Agents | Comparator | Balanced-MSE difference | Bootstrap 95% CI | Peer wins |
|---:|---|---:|---:|---:|
| 1 | fixed_ewc | 0.02913 | [0.01682, 0.04261] | 34/100 |
| 1 | local_adaptive | 0.00000 | [0.00000, 0.00000] | 0/100 |
| 1 | shuffled_communication | -0.04850 | [-0.10617, -0.00426] | 53/100 |
| 2 | fixed_ewc | 0.09739 | [0.07627, 0.11984] | 14/100 |
| 2 | local_adaptive | 0.06772 | [0.04234, 0.09432] | 35/100 |
| 2 | shuffled_communication | 0.02347 | [-0.00826, 0.05495] | 28/100 |
| 4 | fixed_ewc | 0.02509 | [0.01719, 0.03448] | 23/100 |
| 4 | local_adaptive | -0.00064 | [-0.01256, 0.01232] | 61/100 |
| 4 | shuffled_communication | 0.00064 | [-0.01720, 0.01475] | 45/100 |
| 8 | fixed_ewc | 0.01461 | [0.01078, 0.01877] | 23/100 |
| 8 | local_adaptive | -0.01638 | [-0.02416, -0.00828] | 68/100 |
| 8 | shuffled_communication | -0.00130 | [-0.00469, 0.00212] | 58/100 |

## Interpretation

This isolates a small adaptive-consolidation signal in a synthetic problem; it cannot establish MACL performance on LLMs or negotiation. Increasing agent count changes the distribution of agent-specific tasks while keeping the total number of optimizer updates fixed. The one-agent peer-communication arm has no peers and uses its local signal as a documented fallback. Treat trends as exploratory; a favorable result needs replication on disjoint negotiation tasks.

Machine-readable results: [multiagent_scaling_study.json](./multiagent_scaling_study.json).
