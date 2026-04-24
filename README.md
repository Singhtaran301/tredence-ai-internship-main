# Self-Pruning Neural Network

This repository contains my solution for the Tredence AI Engineer case study, "The Self-Pruning Neural Network". The project builds a feed-forward neural network for CIFAR-10 where every weight is paired with a learnable gate. During training, the network learns which connections are less useful and suppresses them through an L1 sparsity penalty applied to the gate values.

## What Is Included

- `solution.py`: training script, custom `PrunableLinear` layer, evaluation pipeline, metrics export, and plot generation
- `app.py`: FastAPI inference service for the trained best model
- `REPORT.md`: short summary tailored to the assignment
- `artifacts_final/`: final trained models, plots, and measured results
- `render.yaml`: simple deployment config for Render

## Approach

Each linear layer is replaced with a custom `PrunableLinear` module. It keeps:

- a standard `weight`
- a standard `bias`
- a learnable `gate_scores` tensor with the same shape as `weight`

In the forward pass:

1. `gate_scores` are passed through a sigmoid to create gates between `0` and `1`
2. the weight matrix is multiplied element-wise by the gates
3. the linear transformation uses the gated weights

The training objective is:

`Total Loss = CrossEntropyLoss + lambda * SparsityLoss`

The sparsity term is a normalized L1-style penalty over all sigmoid gate values. This encourages unimportant gates to move close to zero while preserving important connections for classification performance.

## My Journey

The first version of the project had the right idea, but it was not submission-ready. I found a few issues during review:

- CIFAR-10 normalization was incorrect for RGB images
- the report values did not match the code's lambda settings
- the API expected a model file that was not aligned with the final training outputs
- the repository did not clearly document how to reproduce the results

I then reworked the project to make it consistent end-to-end:

- rebuilt `solution.py` into a proper runnable training pipeline
- added result export to `csv` and `json`
- generated gate distribution plots automatically
- tuned the gate parameterization so sparsity actually shows up under the assignment threshold `gate < 1e-2`
- selected a best model and wired the API to that model
- added deployment configuration and complete usage instructions

## Results

These results were generated with:

```bash
python solution.py --epochs 5 --batch-size 128 --num-workers 0 --lambdas 1 5 10 --output-dir artifacts_final
```

| Lambda | Test Accuracy (%) | Sparsity Level (%) | Pruned Weights | Total Weights |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 52.76 | 2.08 | 35,527 | 1,706,496 |
| 5 | 54.96 | 12.57 | 214,425 | 1,706,496 |
| 10 | 54.06 | 33.28 | 567,864 | 1,706,496 |

Best model:

- selected lambda: `5`
- test accuracy: `54.96%`
- sparsity: `12.57%`
- gate plot: `artifacts_final/plots/gate_distribution_lambda_5.png`
- model weights: `artifacts_final/models/model_lambda_5.pth`

## How To Run On Another Machine

### 1. Clone the repository

```bash
git clone https://github.com/<your-github-username>/<your-repo-name>.git
cd tredence-ai-internship
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training

```bash
python solution.py --epochs 5 --batch-size 128 --num-workers 0 --lambdas 1 5 10 --output-dir artifacts_final
```

This will:

- download CIFAR-10 into `data/`
- train the model for three lambda values
- save trained checkpoints
- save gate distribution plots
- write `results.csv`, `results.json`, and an auto-generated report

### 5. Run the API locally

The API is already configured to use the best saved model at `artifacts_final/models/model_lambda_5.pth`.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

### 6. Test prediction

Use the Swagger UI at `/docs` and upload an image, or call the endpoint directly:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample_image.png"
```

## Deployment Note

This project is better suited to Render or Railway than Vercel because FastAPI with PyTorch is relatively heavy for serverless deployment. A ready-to-use Render configuration is included in `render.yaml`.

## Trade-Offs

This project is built around the main trade-off in self-pruning networks: stronger sparsity pressure can remove more connections, but too much pressure can hurt predictive performance.

- lower `lambda` preserves more weights and usually protects accuracy, but pruning remains limited
- higher `lambda` increases sparsity, but can over-regularize the model and reduce useful capacity
- in my final sweep, `lambda=5` gave the best balance between accuracy and pruning
- `lambda=10` pruned much more aggressively, but accuracy dipped slightly compared with `lambda=5`
- the model is a simple fully connected network, which keeps the implementation clear for the assignment but is not as strong as a modern CNN on CIFAR-10
- sigmoid gates are smooth and trainable, but they usually approach zero rather than becoming mathematically exact zeros, so sparsity is measured with the assignment threshold `gate < 1e-2`

## Files To Review

- `solution.py`
- `app.py`
- `REPORT.md`
- `artifacts_final/results.csv`
- `artifacts_final/plots/gate_distribution_lambda_5.png`

## Assignment Checklist

- custom `PrunableLinear` layer implemented from scratch
- learnable gate tensor registered as parameters
- gated forward pass implemented correctly
- sparsity loss added to the training objective
- CIFAR-10 training and evaluation pipeline included
- results reported for three lambda values
- sparsity percentage computed with a threshold
- plot of final gate distribution saved
- deployment-ready inference API included
