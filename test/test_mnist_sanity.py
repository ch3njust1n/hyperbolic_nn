"""Tests for the MNIST hyperbolic sanity trainer."""

import os
import sys

import torch

DEFAULT_SERVICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "services", "hyperbolic-nn")
)
SERVICE_DIR = os.environ.get("HYPERBOLIC_NN_SERVICE_DIR", DEFAULT_SERVICE_DIR)
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import mnist_sanity


def test_mnist_forward_returns_finite_logits() -> None:
    """MNIST sanity model returns finite logits with shape [batch, 10]."""
    model = mnist_sanity.MnistSanityClassifier(hidden_dim=8, c=1.0, dtype=torch.float64)
    images = torch.rand(4, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (4, 10)
    assert torch.isfinite(logits).all()


def test_mnist_training_step_updates_parameters() -> None:
    """One MNIST sanity training step changes model parameters and returns finite loss."""
    model = mnist_sanity.MnistSanityClassifier(hidden_dim=8, c=1.0, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = torch.rand(8, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    train_loader = mnist_sanity.build_tensor_loader(
        images=images, labels=labels, batch_size=4
    )
    before = model.A_mlr.detach().clone()

    loss = mnist_sanity.train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        max_train_batches=1,
    )

    assert loss > 0.0
    assert not torch.equal(before, model.A_mlr.detach())


def test_save_training_plots_writes_png_files(tmp_path) -> None:
    """Training plot helper writes loss and accuracy PNG files."""
    batch_losses = [2.3, 1.8, 1.2]
    epoch_train_losses = [1.77, 1.2]
    epoch_accuracies = [0.82, 0.91]
    train_loss_path, eval_accuracy_path = mnist_sanity.save_training_plots(
        batch_losses=batch_losses,
        epoch_train_losses=epoch_train_losses,
        epoch_accuracies=epoch_accuracies,
        output_dir=str(tmp_path),
    )
    assert os.path.isfile(train_loss_path)
    assert os.path.isfile(eval_accuracy_path)
