# Self-Pruning Neural Network Report

## Why L1 On Sigmoid Gates Encourages Sparsity
The L1 penalty adds a cost for every active gate, so the optimizer is rewarded for shrinking unimportant gates.
Because each gate is the sigmoid of a learnable score, pushing a score more negative drives the gate closer to 0, which effectively removes that weight from the layer.
In practice, sigmoid gates rarely become mathematically exact zeros, so sparsity is reported using the assignment threshold of gate < 1e-2.

## Results
| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| :--- | ---: | ---: |
| 1 | 52.76 | 2.08 |
| 5 | 54.96 | 12.57 |
| 10 | 54.06 | 33.28 |

## Best Model Gate Distribution
Best model selected by test accuracy: lambda=5.
Gate distribution plot: `artifacts_final\plots\gate_distribution_lambda_5.png`.
