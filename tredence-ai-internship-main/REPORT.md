# Self-Pruning Neural Network Report

## Why L1 On Sigmoid Gates Encourages Sparsity
The sparsity term adds a penalty for every gate that stays active. Since each gate is `sigmoid(gate_score)`, the optimizer can reduce the penalty by pushing unimportant gate scores downward, which moves those gates closer to `0` and weakens the corresponding weights. This creates a direct trade-off between classification performance and keeping connections alive.

In practice, sigmoid gates usually become very small rather than mathematically exact zeros, so the assignment's sparsity metric should be reported with the required threshold check: gate value `< 1e-2`. The training script uses a normalized L1-style penalty so the regularization strength stays numerically stable across all gated layers.

## Results Table
These results were generated with `python solution.py --epochs 5 --batch-size 128 --num-workers 0 --lambdas 1 5 10 --output-dir artifacts_final`.

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| :--- | ---: | ---: |
| 1 | 52.76 | 2.08 |
| 5 | 54.96 | 12.57 |
| 10 | 54.06 | 33.28 |

## Gate Distribution Plot
The best model from this sweep is `lambda=5` because it gave the highest test accuracy while still achieving meaningful pruning. The saved plot is `artifacts_final/plots/gate_distribution_lambda_5.png`.
