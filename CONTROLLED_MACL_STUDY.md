# Controlled synthetic MACL stress test

## Protocol

A two-agent linear-regression continual-learning diagnostic uses 32 validation seeds for hyperparameter selection and 100 separate test seeds. Agents learn Task A, then Task B with varying but correlated task conflict. All methods use identical data, initialization, and 160 gradient steps per task.

Fixed EWC strength and MACL's low/high strength and signal scale are selected by minimum validation mean of Task-A and Task-B test MSE. Both arms receive 108 validation candidates. The partner-signal MACL arm is compared to fixed EWC, naive learning, a self-signal ablation, and a shuffled-partner-signal ablation. The reported 95% intervals are bootstrap intervals over test seeds.

Selected fixed EWC lambda: `1.0101193669796606`.
Selected MACL parameters: low=`0.3`, high=`1.5`, signal scale=`1.0`.

## Held-out test results

Lower is better for all losses. Retention change is Task-A test MSE after B minus after A; a smaller increase indicates less forgetting.

| Method | Balanced MSE | Task-A retention change | Task-A MSE after B | Task-B MSE after B |
|---|---:|---:|---:|---:|
| Naive | 7.76108 +/- 4.29005 [6.93204, 8.59267] | 15.50850 +/- 8.58013 [13.85042, 17.17169] | 15.51530 +/- 8.58009 [13.85720, 17.17847] | 0.00686 +/- 0.00034 [0.00679, 0.00692] |
| Validation-tuned fixed EWC | 3.97095 +/- 2.19306 [3.54584, 4.40115] | 3.93778 +/- 2.21910 [3.50036, 4.36763] | 3.94458 +/- 2.21907 [3.50711, 4.37447] | 3.99732 +/- 2.27255 [3.55207, 4.44539] |
| MACL (partner signal) | 4.00720 +/- 2.19513 [3.58179, 4.43779] | 3.65998 +/- 1.65453 [3.33522, 3.97988] | 3.66678 +/- 1.65451 [3.34201, 3.98662] | 4.34761 +/- 2.88989 [3.78421, 4.92025] |
| MACL self-signal ablation | 4.00110 +/- 2.20625 [3.57362, 4.43412] | 3.57900 +/- 1.68511 [3.24696, 3.90496] | 3.58581 +/- 1.68509 [3.25372, 3.91177] | 4.41640 +/- 2.85823 [3.86030, 4.98022] |
| MACL shuffled-signal ablation | 4.04911 +/- 2.25376 [3.61145, 4.48721] | 4.16131 +/- 2.73052 [3.63238, 4.69819] | 4.16811 +/- 2.73054 [3.63916, 4.70498] | 3.93011 +/- 2.50806 [3.45201, 4.42093] |

Paired MACL-minus-fixed-EWC balanced test MSE: 0.036246 +/- 0.061604; bootstrap 95% CI [0.024740, 0.048970]. MACL has lower balanced MSE on 24/100 test seeds.

The partner-signal arm is compared directly to both its own-signal ablation and a shuffled-partner-signal ablation. Lower paired differences favor partner communication:
- `macl_minus_self_signal_balanced_mse`: 0.006092, bootstrap 95% CI [-0.002351, 0.016711], MACL is lower on 48/100 seeds.
- `macl_minus_self_signal_task_a_retention_change`: 0.080976, bootstrap 95% CI [0.045201, 0.127672], MACL is lower on 20/100 seeds.
- `macl_minus_shuffled_balanced_mse`: -0.041917, bootstrap 95% CI [-0.085819, -0.005278], MACL is lower on 50/100 seeds.
- `macl_minus_fixed_task_a_retention_change`: -0.277799, bootstrap 95% CI [-0.404672, -0.150324], MACL is lower on 59/100 seeds.
- `macl_minus_fixed_task_b_mse`: 0.350290, bootstrap 95% CI [0.225531, 0.478264], MACL is lower on 41/100 seeds.

## Interpretation

This synthetic experiment tests a simplified adaptive-EWC mechanism, not the full MACL algorithm, language models, or negotiation. A result in this benchmark cannot prove general MACL superiority. A MACL win must also preserve or improve Task-B performance, outperform both self-signal and shuffled-signal ablations, and be replicated on the disjoint-prompt LLM study before supporting a broader claim.

Machine-readable full results: [controlled_macl_study.json](./controlled_macl_study.json).
