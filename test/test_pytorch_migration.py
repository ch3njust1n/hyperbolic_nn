"""Permanent tests for the PyTorch hyperbolic RNN migration."""

import os
import sys

import numpy as np
import torch

DEFAULT_SERVICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "services", "hyperbolic-nn")
)
SERVICE_DIR = os.environ.get("HYPERBOLIC_NN_SERVICE_DIR", DEFAULT_SERVICE_DIR)
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import hyp_rnn
import util


def test_mobius_torch_matches_numpy() -> None:
    """PyTorch Mobius addition matches the NumPy reference for shape [batch, dim]."""
    c = 1.0
    first_np = np.array([[0.01, -0.02, 0.03]], dtype=np.float64)
    second_np = np.array([[0.03, 0.01, -0.02]], dtype=np.float64)
    first_torch = torch.tensor(first_np, dtype=torch.float64)
    second_torch = torch.tensor(second_np, dtype=torch.float64)

    actual = util.mob_add(first_torch, second_torch, c).detach().cpu().numpy()[0]
    expected = util.mob_add_np(first_np[0], second_np[0], c)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_next_batch_wraps_final_partial_batch() -> None:
    """Final partial batches wrap to a fixed batch shape [batch_size, max_tokens]."""
    data = [
        ([1, 2], 2, [3], 1, 1),
        ([4], 1, [5, 6, 7], 3, 0),
        ([8, 9, 10, 11], 4, [12], 1, 1),
    ]
    batch = hyp_rnn.next_batch(i=2, batch_size=2, data=data)

    assert batch.word_ids_1.shape == (2, 4)
    assert batch.word_ids_2.shape == (2, 4)
    assert batch.labels.tolist() == [1, 1]


def test_forward_pass_returns_finite_loss() -> None:
    """The model computes finite logits and loss for one PRFX10 batch."""
    config = hyp_rnn.TrainingConfig(
        root_path=SERVICE_DIR,
        dataset="PRFX10",
        batch_size=2,
        word_dim=5,
        before_mlr_dim=5,
        inputs_geom="hyp",
        sent_geom="hyp",
        bias_geom="hyp",
        ffnn_geom="hyp",
        mlr_geom="hyp",
        cell_type="gru",
        additional_features="dsq",
    )
    word_to_id, id_to_word = hyp_rnn.load_vocabulary(config)
    train_data, _, _ = hyp_rnn.load_datasets(config)
    batch = hyp_rnn.next_batch(i=0, batch_size=config.batch_size, data=train_data[:2])
    model = hyp_rnn.HyperbolicRNNModel(
        config=config, word_to_id=word_to_id, id_to_word=id_to_word
    )

    logits, distance_sq = model(batch, dropout_keep_prob=1.0)
    loss = hyp_rnn.compute_loss(
        config=config, logits=logits, labels=batch.labels, distance_sq=distance_sq
    )

    assert logits.shape == (2, 2)
    assert torch.isfinite(loss)


def test_one_training_step_updates_and_projects() -> None:
    """One optimizer step updates parameters and keeps hyperbolic tensors inside the ball."""
    config = hyp_rnn.TrainingConfig(
        root_path=SERVICE_DIR,
        dataset="PRFX10",
        batch_size=2,
        word_dim=5,
        before_mlr_dim=5,
        inputs_geom="hyp",
        sent_geom="hyp",
        bias_geom="hyp",
        ffnn_geom="hyp",
        mlr_geom="hyp",
        cell_type="gru",
        additional_features="dsq",
    )
    word_to_id, id_to_word = hyp_rnn.load_vocabulary(config)
    train_data, _, _ = hyp_rnn.load_datasets(config)
    batch = hyp_rnn.next_batch(i=0, batch_size=config.batch_size, data=train_data[:2])
    model = hyp_rnn.HyperbolicRNNModel(
        config=config, word_to_id=word_to_id, id_to_word=id_to_word
    )
    trainer = hyp_rnn.RiemannianTrainer(config=config, model=model)
    before = model.embeddings.detach().clone()

    loss = trainer.train_batch(batch=batch, burn_in_factor=1.0)

    assert torch.isfinite(torch.tensor(loss, dtype=torch.float64))
    assert not torch.equal(before, model.embeddings.detach())
    assert trainer.max_hyperbolic_norm() < (1.0 - util.PROJ_EPS) / np.sqrt(config.c)
