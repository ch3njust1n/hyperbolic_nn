"""Train a small hyperbolic MNIST classifier as a PyTorch sanity check."""

import argparse
import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

import util


@dataclass
class MnistConfig:
    """Configuration for the MNIST sanity run."""

    data_dir: str
    output_dir: str
    batch_size: int
    hidden_dim: int
    learning_rate: float
    epochs: int
    max_train_batches: int
    max_eval_batches: int
    train_samples: int
    test_samples: int
    c: float
    seed: int
    device: str
    download: bool


class MnistSanityClassifier(nn.Module):
    """Small MNIST classifier with a hyperbolic final representation.

    Inputs:
        images: Tensor with shape [batch, 1, 28, 28].

    Returns:
        Logits tensor with shape [batch, 10].
    """

    def __init__(self, hidden_dim: int, c: float, dtype: torch.dtype) -> None:
        """Initialize Euclidean projection and hyperbolic MLR parameters."""
        super().__init__()
        self.c = c
        self.dtype = dtype
        self.input_projection = nn.Linear(28 * 28, hidden_dim, dtype=dtype)
        self.A_mlr = nn.Parameter(torch.empty(10, hidden_dim, dtype=dtype))
        self.P_mlr = nn.Parameter(torch.zeros(10, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.A_mlr)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute MNIST class logits for images with shape [batch, 1, 28, 28]."""
        flat_images = images.reshape(images.shape[0], 28 * 28).to(dtype=self.dtype)
        euclidean_features = torch.tanh(self.input_projection(flat_images))
        hyperbolic_features = util.exp_map_zero(euclidean_features, self.c)
        logits_list = []
        for class_index in range(10):
            class_a = self.A_mlr[class_index : class_index + 1]
            class_p = self.P_mlr[class_index : class_index + 1]
            minus_p_plus_x = util.mob_add(-class_p, hyperbolic_features, self.c)
            norm_a = util.norm(class_a)
            lambda_px = util.lambda_x(minus_p_plus_x, self.c)
            px_dot_a = util.dot(minus_p_plus_x, F.normalize(class_a, p=2, dim=1))
            logit = (
                2.0
                / np.sqrt(self.c)
                * norm_a
                * torch.asinh(np.sqrt(self.c) * px_dot_a * lambda_px)
            )
            logits_list.append(logit.reshape(-1))
        return torch.stack(logits_list, dim=1)

    def project_hyperbolic_parameters(self) -> None:
        """Project hyperbolic MLR points back into the Poincare ball."""
        with torch.no_grad():
            self.P_mlr.data.copy_(util.project_hyp_vecs(self.P_mlr.data, self.c))


def str2bool(answer: str) -> bool:
    """Parse y/n CLI values as booleans."""
    lowered_answer = answer.lower()
    if lowered_answer in ["y", "yes", "true"]:
        return True
    if lowered_answer in ["n", "no", "false"]:
        return False
    raise ValueError("Invalid boolean answer: " + answer)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the MNIST sanity run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/jobs/datasets")
    parser.add_argument("--output_dir", type=str, default="/jobs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument(
        "--train_samples",
        type=int,
        default=0,
        help="Number of train examples; 0 uses the full MNIST train split.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=0,
        help="Number of test examples; 0 uses the full MNIST test split.",
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--download", type=str2bool, default=True)
    return parser


def config_from_args(args: argparse.Namespace) -> MnistConfig:
    """Convert CLI args into a typed MNIST config."""
    return MnistConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        c=args.c,
        seed=args.seed,
        device=args.device,
        download=args.download,
    )


def resolve_device(device: str) -> torch.device:
    """Resolve requested device or use CUDA when available."""
    if device != "":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_mnist_loaders(config: MnistConfig) -> tuple[DataLoader, DataLoader]:
    """Create train/test loaders from torchvision MNIST datasets."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, download=config.download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, download=config.download, transform=transform
    )
    train_count = (
        len(train_dataset) if config.train_samples == 0 else config.train_samples
    )
    test_count = len(test_dataset) if config.test_samples == 0 else config.test_samples
    train_subset = Subset(train_dataset, list(range(train_count)))
    test_subset = Subset(test_dataset, list(range(test_count)))
    train_loader = DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_subset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def build_tensor_loader(
    images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    """Create a loader from tensors for tests, with images [n, 1, 28, 28]."""
    return DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=False
    )


def train_one_epoch(
    model: MnistSanityClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_train_batches: int,
    batch_losses: list[float] | None = None,
) -> float:
    """Train for one epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        if max_train_batches > 0 and batch_index >= max_train_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        model.project_hyperbolic_parameters()
        batch_loss = float(loss.detach().cpu().item())
        total_loss += batch_loss
        num_batches += 1
        if batch_losses is not None:
            batch_losses.append(batch_loss)
    if num_batches == 0:
        raise ValueError("No MNIST training batches were processed")
    return total_loss / float(num_batches)


def evaluate(
    model: MnistSanityClassifier,
    test_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int,
) -> float:
    """Evaluate accuracy on MNIST batches."""
    model.eval()
    num_correct = 0
    num_examples = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            if max_eval_batches > 0 and batch_index >= max_eval_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            predictions = torch.argmax(model(images), dim=1)
            num_correct += int(torch.sum(predictions == labels).detach().cpu().item())
            num_examples += int(labels.shape[0])
    if num_examples == 0:
        raise ValueError("No MNIST evaluation batches were processed")
    return num_correct / float(num_examples)


def save_training_plots(
    batch_losses: list[float],
    epoch_train_losses: list[float],
    epoch_accuracies: list[float],
    output_dir: str,
) -> tuple[str, str]:
    """Save training loss and eval accuracy plots; returns PNG paths."""
    os.makedirs(output_dir, exist_ok=True)
    train_loss_path = os.path.join(output_dir, "mnist_sanity_train_loss.png")
    eval_accuracy_path = os.path.join(output_dir, "mnist_sanity_eval_accuracy.png")
    batch_steps = list(range(1, len(batch_losses) + 1))
    epoch_steps = list(range(1, len(epoch_accuracies) + 1))

    train_figure, train_axis = plt.subplots(figsize=(10, 5))
    train_axis.plot(
        batch_steps, batch_losses, color="#2563eb", linewidth=1.0, alpha=0.85
    )
    train_axis.set_title("MNIST Sanity Training Loss")
    train_axis.set_xlabel("Training batch")
    train_axis.set_ylabel("Cross-entropy loss")
    train_axis.grid(True, alpha=0.3)
    train_figure.tight_layout()
    train_figure.savefig(train_loss_path, dpi=150)
    plt.close(train_figure)

    accuracy_figure, accuracy_axis = plt.subplots(figsize=(8, 5))
    accuracy_axis.plot(
        epoch_steps,
        epoch_accuracies,
        color="#16a34a",
        marker="o",
        linewidth=2.0,
    )
    accuracy_axis.set_title("MNIST Sanity Eval Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Test accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_figure.tight_layout()
    accuracy_figure.savefig(eval_accuracy_path, dpi=150)
    plt.close(accuracy_figure)

    return train_loss_path, eval_accuracy_path


def run_training(config: MnistConfig) -> tuple[float, float]:
    """Train the MNIST sanity model and return final loss and accuracy."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    device = resolve_device(config.device)
    train_loader, test_loader = build_mnist_loaders(config)
    model = MnistSanityClassifier(
        hidden_dim=config.hidden_dim, c=config.c, dtype=torch.float64
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    final_loss = 0.0
    batch_losses: list[float] = []
    epoch_train_losses: list[float] = []
    epoch_accuracies: list[float] = []
    start_time = time.perf_counter()
    for epoch_index in range(config.epochs):
        final_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_train_batches=config.max_train_batches,
            batch_losses=batch_losses,
        )
        epoch_train_losses.append(final_loss)
        epoch_accuracy = evaluate(
            model=model,
            test_loader=test_loader,
            device=device,
            max_eval_batches=config.max_eval_batches,
        )
        epoch_accuracies.append(epoch_accuracy)
        print(
            "epoch="
            + str(epoch_index + 1)
            + "/"
            + str(config.epochs)
            + " loss="
            + format(final_loss, ".4f")
            + " test_accuracy="
            + format(epoch_accuracy, ".4f")
        )
    accuracy = evaluate(
        model=model,
        test_loader=test_loader,
        device=device,
        max_eval_batches=config.max_eval_batches,
    )
    elapsed_seconds = time.perf_counter() - start_time
    checkpoint_path = os.path.join(config.output_dir, "mnist_sanity.pt")
    metrics_path = os.path.join(config.output_dir, "mnist_sanity_metrics.json")
    train_loss_plot_path, eval_accuracy_plot_path = save_training_plots(
        batch_losses=batch_losses,
        epoch_train_losses=epoch_train_losses,
        epoch_accuracies=epoch_accuracies,
        output_dir=config.output_dir,
    )
    metrics_payload = {
        "batch_losses": batch_losses,
        "epoch_train_losses": epoch_train_losses,
        "epoch_accuracies": epoch_accuracies,
        "final_loss": final_loss,
        "final_accuracy": accuracy,
        "elapsed_seconds": elapsed_seconds,
    }
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)
    torch.save(model.state_dict(), checkpoint_path)
    print(
        "MNIST sanity complete: "
        + "loss="
        + format(final_loss, ".4f")
        + " accuracy="
        + format(accuracy, ".4f")
        + " seconds="
        + format(elapsed_seconds, ".2f")
        + " checkpoint="
        + checkpoint_path
        + " train_loss_plot="
        + train_loss_plot_path
        + " eval_accuracy_plot="
        + eval_accuracy_plot_path
    )
    return final_loss, accuracy


def run() -> None:
    """Run the MNIST sanity CLI."""
    parser = build_arg_parser()
    config = config_from_args(parser.parse_args())
    run_training(config)


if __name__ == "__main__":
    run()
