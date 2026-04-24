import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrunableLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.temperature = 0.1
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.gate_scores, mean=0.5, std=0.1)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores / self.temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weights = self.weight * self.gates()
        return F.linear(x, pruned_weights, self.bias)


class SelfPruningNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sparsity_loss(self) -> torch.Tensor:
        penalties = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                penalties.append(module.gates().mean())
        return torch.stack(penalties).mean()

    def gate_values(self) -> torch.Tensor:
        gate_chunks = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gate_chunks.append(module.gates().reshape(-1))
        return torch.cat(gate_chunks)


@dataclass
class RunResult:
    lambda_value: float
    test_accuracy: float
    sparsity_percent: float
    pruned_weights: int
    total_weights: int
    model_path: str
    plot_path: str


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate(model: SelfPruningNet, data_loader: DataLoader, device: torch.device, threshold: float) -> tuple[float, float, int, int, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    gates = model.gate_values().detach().cpu()
    pruned = int((gates < threshold).sum().item())
    total_weights = gates.numel()
    accuracy = 100.0 * correct / total
    sparsity = 100.0 * pruned / total_weights
    return accuracy, sparsity, pruned, total_weights, gates.numpy()


def plot_gate_distribution(gates: np.ndarray, lambda_value: float, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=60, color="#1f77b4", edgecolor="black")
    plt.title(f"Gate Distribution (lambda={lambda_value:g})")
    plt.xlabel("Gate value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_one_lambda(
    lambda_value: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    threshold: float,
    output_dir: Path,
) -> RunResult:
    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            classification_loss = criterion(logits, labels)
            sparsity_penalty = model.sparsity_loss()
            total_loss = classification_loss + lambda_value * sparsity_penalty
            total_loss.backward()
            optimizer.step()

        print(f"lambda={lambda_value:g} epoch={epoch + 1}/{epochs} completed")

    accuracy, sparsity, pruned, total_weights, gates = evaluate(model, test_loader, device, threshold)

    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"model_lambda_{lambda_value:g}.pth"
    plot_path = plots_dir / f"gate_distribution_lambda_{lambda_value:g}.png"

    torch.save(model.state_dict(), model_path)
    plot_gate_distribution(gates, lambda_value, plot_path)

    return RunResult(
        lambda_value=lambda_value,
        test_accuracy=accuracy,
        sparsity_percent=sparsity,
        pruned_weights=pruned,
        total_weights=total_weights,
        model_path=str(model_path),
        plot_path=str(plot_path),
    )


def write_results_summary(results: list[RunResult], output_dir: Path) -> tuple[Path, Path]:
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "lambda_value",
                "test_accuracy",
                "sparsity_percent",
                "pruned_weights",
                "total_weights",
                "model_path",
                "plot_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(result) for result in results], json_file, indent=2)

    return csv_path, json_path


def write_report_template(results: list[RunResult], output_dir: Path) -> Path:
    best_result = max(results, key=lambda item: item.test_accuracy)
    report_path = output_dir / "REPORT.md"

    lines = [
        "# Self-Pruning Neural Network Report",
        "",
        "## Why L1 On Sigmoid Gates Encourages Sparsity",
        "The L1 penalty adds a cost for every active gate, so the optimizer is rewarded for shrinking unimportant gates.",
        "Because each gate is the sigmoid of a learnable score, pushing a score more negative drives the gate closer to 0, which effectively removes that weight from the layer.",
        "In practice, sigmoid gates rarely become mathematically exact zeros, so sparsity is reported using the assignment threshold of gate < 1e-2.",
        "",
        "## Results",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "| :--- | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            f"| {result.lambda_value:g} | {result.test_accuracy:.2f} | {result.sparsity_percent:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Best Model Gate Distribution",
            f"Best model selected by test accuracy: lambda={best_result.lambda_value:g}.",
            f"Gate distribution plot: `{best_result.plot_path}`.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-pruning neural network on CIFAR-10.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    results = []
    for lambda_value in args.lambdas:
        result = train_one_lambda(
            lambda_value=lambda_value,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            threshold=args.threshold,
            output_dir=output_dir,
        )
        results.append(result)
        print(
            f"lambda={result.lambda_value:g} "
            f"accuracy={result.test_accuracy:.2f}% "
            f"sparsity={result.sparsity_percent:.2f}%"
        )

    csv_path, json_path = write_results_summary(results, output_dir)
    report_path = write_report_template(results, output_dir)

    print(f"Saved results to {csv_path}")
    print(f"Saved results to {json_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
