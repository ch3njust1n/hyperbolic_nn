import argparse
import os
import pickle
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import rnn_impl
import util

SentencePair = tuple[list[int], int, list[int], int, int]
DatasetSplits = tuple[list[SentencePair], list[SentencePair], list[SentencePair]]
Vocabulary = tuple[dict[str, int], dict[int, str]]


@dataclass
class TrainingConfig:
    """Configuration for training and evaluation."""

    base_name: str = ""
    root_path: str = "/path/to/your/data/folders/"
    output_dir: str = ""
    dataset: str = "SNLI"
    word_dim: int = 5
    word_init_avg_norm: float = 0.001
    inputs_geom: str = "eucl"
    cell_type: str = "rnn"
    cell_non_lin: str = "id"
    sent_geom: str = "eucl"
    bias_geom: str = "eucl"
    fix_biases: bool = False
    fix_matrices: bool = False
    matrices_init_eye: bool = False
    before_mlr_dim: int = 5
    ffnn_non_lin: str = "id"
    ffnn_geom: str = "eucl"
    additional_features: str = ""
    dropout: float = 1.0
    mlr_geom: str = "eucl"
    proj_eps: float = 1e-5
    reg_beta: float = 0.0
    hyp_opt: str = "rsgd"
    lr_ffnn: float = 0.01
    lr_words: float = 0.1
    batch_size: int = 64
    burnin: bool = False
    c: float = 1.0
    restore_model: bool = False
    restore_from_path: str = ""
    num_epochs: int = 30
    num_classes: int = 2
    print_step: int = 2000
    max_train_batches: int = 0
    max_eval_batches: int = 0
    device: str = ""


@dataclass
class Batch:
    """One padded minibatch of sentence-pair examples."""

    word_ids_1: torch.Tensor
    num_words_1: torch.Tensor
    word_ids_2: torch.Tensor
    num_words_2: torch.Tensor
    labels: torch.Tensor

    def to_device(self, device: torch.device) -> "Batch":
        """Move all tensors to the requested device."""
        return Batch(
            word_ids_1=self.word_ids_1.to(device),
            num_words_1=self.num_words_1.to(device),
            word_ids_2=self.word_ids_2.to(device),
            num_words_2=self.num_words_2.to(device),
            labels=self.labels.to(device),
        )


def str2bool(answer: str) -> bool:
    """Parse y/n CLI values as booleans."""
    lowered_answer = answer.lower()
    if lowered_answer in ["y", "yes", "true"]:
        return True
    if lowered_answer in ["n", "no", "false"]:
        return False
    raise ValueError("Invalid boolean answer: " + answer)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str, default="")
    parser.add_argument("--root_path", type=str, default="/path/to/your/data/folders/")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="SNLI")
    parser.add_argument("--word_dim", type=int, default=5)
    parser.add_argument("--word_init_avg_norm", type=float, default=0.001)
    parser.add_argument("--inputs_geom", type=str, default="eucl")
    parser.add_argument("--cell_type", type=str, default="rnn")
    parser.add_argument("--cell_non_lin", type=str, default="id")
    parser.add_argument("--sent_geom", type=str, default="eucl")
    parser.add_argument("--bias_geom", type=str, default="eucl")
    parser.add_argument("--fix_biases", type=str2bool, default=False)
    parser.add_argument("--fix_matrices", type=str2bool, default=False)
    parser.add_argument("--matrices_init_eye", type=str2bool, default=False)
    parser.add_argument("--before_mlr_dim", type=int, default=5)
    parser.add_argument("--ffnn_non_lin", type=str, default="id")
    parser.add_argument("--ffnn_geom", type=str, default="eucl")
    parser.add_argument("--additional_features", type=str, default="")
    parser.add_argument("--dropout", type=float, default=1.0)
    parser.add_argument("--mlr_geom", type=str, default="eucl")
    parser.add_argument("--proj_eps", type=float, default=1e-5)
    parser.add_argument("--reg_beta", type=float, default=0.0)
    parser.add_argument("--hyp_opt", type=str, default="rsgd")
    parser.add_argument("--lr_ffnn", type=float, default=0.01)
    parser.add_argument("--lr_words", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--burnin", type=str2bool, default=False)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--restore_model", type=str2bool, default=False)
    parser.add_argument("--restore_from_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--print_step", type=int, default=2000)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Convert parsed CLI args into a typed training config."""
    output_dir = args.output_dir
    if output_dir == "":
        output_dir = os.environ.get("OUTPUT_DIR", args.root_path)
    config = TrainingConfig(
        base_name=args.base_name,
        root_path=args.root_path,
        output_dir=output_dir,
        dataset=args.dataset,
        word_dim=args.word_dim,
        word_init_avg_norm=args.word_init_avg_norm,
        inputs_geom=args.inputs_geom,
        cell_type=args.cell_type,
        cell_non_lin=args.cell_non_lin,
        sent_geom=args.sent_geom,
        bias_geom=args.bias_geom,
        fix_biases=args.fix_biases,
        fix_matrices=args.fix_matrices,
        matrices_init_eye=args.matrices_init_eye,
        before_mlr_dim=args.before_mlr_dim,
        ffnn_non_lin=args.ffnn_non_lin,
        ffnn_geom=args.ffnn_geom,
        additional_features=args.additional_features,
        dropout=args.dropout,
        mlr_geom=args.mlr_geom,
        proj_eps=args.proj_eps,
        reg_beta=args.reg_beta,
        hyp_opt=args.hyp_opt,
        lr_ffnn=args.lr_ffnn,
        lr_words=args.lr_words,
        batch_size=args.batch_size,
        burnin=args.burnin,
        c=args.c,
        restore_model=args.restore_model,
        restore_from_path=args.restore_from_path,
        num_epochs=args.num_epochs,
        print_step=args.print_step,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
    )
    validate_config(config)
    return config


def validate_config(config: TrainingConfig) -> None:
    """Validate geometry and optimizer combinations before model construction."""
    if config.dataset not in ["SNLI", "PRFX10", "PRFX30", "PRFX50"]:
        raise ValueError("Invalid dataset: " + config.dataset)
    if config.additional_features not in ["", "dsq"]:
        raise ValueError("Invalid additional_features: " + config.additional_features)
    if config.hyp_opt not in ["rsgd", "projsgd"]:
        raise ValueError("Invalid hyp_opt: " + config.hyp_opt)
    for geometry_name in [
        config.sent_geom,
        config.inputs_geom,
        config.bias_geom,
        config.mlr_geom,
        config.ffnn_geom,
    ]:
        if geometry_name not in ["eucl", "hyp"]:
            raise ValueError("Invalid geometry: " + geometry_name)
    if config.cell_type not in ["rnn", "gru", "TFrnn", "TFgru", "TFlstm"]:
        raise ValueError("Invalid cell_type: " + config.cell_type)
    if config.sent_geom == "eucl" and (
        config.inputs_geom != "eucl"
        or config.bias_geom != "eucl"
        or config.ffnn_geom != "eucl"
        or config.mlr_geom != "eucl"
    ):
        raise ValueError(
            "Euclidean sentence geometry requires all downstream geometries to be euclidean"
        )
    if config.ffnn_geom == "hyp" and config.sent_geom != "hyp":
        raise ValueError("Hyperbolic FFNN requires hyperbolic sentence geometry")
    if config.ffnn_geom == "eucl" and config.mlr_geom != "eucl":
        raise ValueError("Euclidean FFNN requires euclidean MLR")
    if config.mlr_geom == "hyp" and (
        config.ffnn_geom != "hyp" or config.sent_geom != "hyp"
    ):
        raise ValueError(
            "Hyperbolic MLR requires hyperbolic FFNN and sentence geometry"
        )
    if config.restore_model and config.restore_from_path == "":
        raise ValueError("restore_from_path is required when restore_model is true")


def resolve_device(config: TrainingConfig) -> torch.device:
    """Resolve the torch device for tensors and modules."""
    if config.device != "":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dataset_paths(config: TrainingConfig) -> tuple[str, str, str, str, str]:
    """Return vocabulary and split paths for the configured dataset."""
    if config.dataset.startswith("PRFX"):
        suffix_by_dataset = {
            "PRFX10": "prefix_10_dataset",
            "PRFX30": "prefix_30_dataset",
            "PRFX50": "prefix_50_dataset",
        }
        suffix = suffix_by_dataset[config.dataset]
        word_to_id_file_path = os.path.join(config.root_path, suffix, "word_to_id")
        id_to_word_file_path = os.path.join(config.root_path, suffix, "id_to_word")
        training_data_file_path = os.path.join(config.root_path, suffix, "train")
        test_data_file_path = os.path.join(config.root_path, suffix, "test")
        dev_data_file_path = os.path.join(config.root_path, suffix, "dev")
        return (
            word_to_id_file_path,
            id_to_word_file_path,
            training_data_file_path,
            dev_data_file_path,
            test_data_file_path,
        )
    suffix = "_" + str(config.num_classes) + "class"
    word_to_id_file_path = os.path.join(config.root_path, "snli_dataset", "word_to_id")
    id_to_word_file_path = os.path.join(config.root_path, "snli_dataset", "id_to_word")
    training_data_file_path = os.path.join(
        config.root_path, "snli_dataset", "train" + suffix
    )
    test_data_file_path = os.path.join(
        config.root_path, "snli_dataset", "test" + suffix
    )
    dev_data_file_path = os.path.join(config.root_path, "snli_dataset", "dev" + suffix)
    return (
        word_to_id_file_path,
        id_to_word_file_path,
        training_data_file_path,
        dev_data_file_path,
        test_data_file_path,
    )


def load_pickle(file_path: str) -> object:
    """Load a pickle file from an explicit path."""
    with open(file_path, "rb") as pickle_file:
        return pickle.load(pickle_file)


def load_vocabulary(config: TrainingConfig) -> Vocabulary:
    """Load word/id vocabularies for the configured dataset."""
    word_to_id_file_path, id_to_word_file_path, _, _, _ = dataset_paths(config)
    word_to_id = load_pickle(word_to_id_file_path)
    id_to_word = load_pickle(id_to_word_file_path)
    return word_to_id, id_to_word


def load_datasets(config: TrainingConfig) -> DatasetSplits:
    """Load train, dev, and test split lists from pickled files."""
    _, _, training_data_file_path, dev_data_file_path, test_data_file_path = (
        dataset_paths(config)
    )
    training_data = load_pickle(training_data_file_path)
    dev_data = load_pickle(dev_data_file_path)
    test_data = load_pickle(test_data_file_path)
    return training_data, dev_data, test_data


def next_batch(i: int, batch_size: int, data: Sequence[SentencePair]) -> Batch:
    """Create one padded fixed-size batch from sentence-pair data.

    Args:
        i: Start index into `data`.
        batch_size: Number of examples in the returned batch.
        data: Sequence of tuples `(ids1, n1, ids2, n2, label)`.

    Returns:
        Batch tensors with word id shapes [batch_size, max_tokens].
    """
    if len(data) == 0:
        raise ValueError("Cannot create a batch from an empty dataset")
    item_index = i
    stop_index = min(i + batch_size, len(data))
    max_pad = 0
    batch_word_ids_1 = []
    batch_num_words_1 = []
    batch_word_ids_2 = []
    batch_num_words_2 = []
    batch_label = []
    while item_index < stop_index:
        word_ids_1, num_words_1, word_ids_2, num_words_2, label = data[item_index]
        copied_word_ids_1 = list(word_ids_1)
        copied_word_ids_2 = list(word_ids_2)
        max_pad = max(max_pad, len(copied_word_ids_1), len(copied_word_ids_2))
        batch_word_ids_1.append(copied_word_ids_1)
        batch_num_words_1.append(num_words_1)
        batch_word_ids_2.append(copied_word_ids_2)
        batch_num_words_2.append(num_words_2)
        batch_label.append(label)
        item_index += 1
        if item_index == stop_index and stop_index == len(data):
            item_index = 0
            stop_index = batch_size - len(batch_word_ids_1)
    for batch_index in range(batch_size):
        while len(batch_word_ids_1[batch_index]) < max_pad:
            batch_word_ids_1[batch_index].append(0)
        while len(batch_word_ids_2[batch_index]) < max_pad:
            batch_word_ids_2[batch_index].append(0)
        if (
            len(batch_word_ids_1[batch_index]) != max_pad
            or len(batch_word_ids_2[batch_index]) != max_pad
        ):
            raise ValueError("Padded batch has inconsistent sequence lengths")
    return Batch(
        word_ids_1=torch.tensor(batch_word_ids_1, dtype=torch.long),
        num_words_1=torch.tensor(batch_num_words_1, dtype=torch.long),
        word_ids_2=torch.tensor(batch_word_ids_2, dtype=torch.long),
        num_words_2=torch.tensor(batch_num_words_2, dtype=torch.long),
        labels=torch.tensor(batch_label, dtype=torch.long),
    )


def dataset_to_minibatches(
    dataset: Sequence[SentencePair], batch_size: int
) -> list[Batch]:
    """Convert a dataset sequence into fixed-size minibatches."""
    batches_list = []
    item_index = 0
    while item_index < len(dataset):
        batches_list.append(
            next_batch(i=item_index, batch_size=batch_size, data=dataset)
        )
        item_index += batch_size
    return batches_list


def make_experiment_name(config: TrainingConfig) -> str:
    """Build the experiment name used for logs, TensorBoard, and checkpoints."""
    now = datetime.now()
    fix_biases_str = "FIX" if config.fix_biases else ""
    mat_str = ""
    if config.fix_matrices or config.matrices_init_eye:
        mat_str = "WFIXeye" if config.fix_matrices else "Weye"
    hyp_opt_str = ""
    if (
        config.inputs_geom == "hyp"
        or config.bias_geom == "hyp"
        or config.ffnn_geom == "hyp"
        or config.mlr_geom == "hyp"
    ):
        hyp_opt_str = (
            config.hyp_opt
            + "_lrW"
            + str(config.lr_words)
            + "_lrFF"
            + str(config.lr_ffnn)
            + "_"
        )
    c_str = "C" + str(config.c) + "_" if config.c != 1.0 else ""
    drp_str = "drp" + str(config.dropout) + "_" if config.dropout != 1.0 else ""
    burnin_str = "burn" + str(config.burnin).lower() if config.burnin else ""
    reg_beta_str = "reg" + str(config.reg_beta) + "_" if config.reg_beta > 0.0 else ""
    additional_features_str = (
        config.additional_features + "_" if config.additional_features != "" else ""
    )
    return (
        config.base_name
        + "_"
        + config.dataset
        + "_W"
        + str(config.word_dim)
        + "d,"
        + str(config.word_init_avg_norm)
        + "init_"
        + config.cell_type
        + "_cellNonL"
        + config.cell_non_lin
        + "_SENT"
        + config.sent_geom
        + "_INP"
        + config.inputs_geom
        + "_BIAS"
        + config.bias_geom
        + fix_biases_str
        + "_"
        + mat_str
        + "FFNN"
        + config.ffnn_geom
        + str(config.before_mlr_dim)
        + config.ffnn_non_lin
        + "_"
        + additional_features_str
        + drp_str
        + "MLR"
        + config.mlr_geom
        + "_"
        + reg_beta_str
        + hyp_opt_str
        + c_str
        + "prje"
        + str(config.proj_eps)
        + "_bs"
        + str(config.batch_size)
        + "_"
        + burnin_str
        + "__"
        + now.strftime("%H:%M:%S,%dM")
    )


def _initial_embedding(
    vocab_size: int, config: TrainingConfig, dtype: torch.dtype
) -> nn.Parameter:
    """Initialize word embeddings near zero with shape [vocab_size, word_dim]."""
    maxval = (3.0 * (config.word_init_avg_norm**2) / (2.0 * config.word_dim)) ** (
        1.0 / 3.0
    )
    embeddings = torch.empty(vocab_size, config.word_dim, dtype=dtype).uniform_(
        -maxval, maxval
    )
    return nn.Parameter(embeddings)


class HyperbolicRNNModel(nn.Module):
    """Sentence-pair classifier with Euclidean or hyperbolic recurrent encoders."""

    def __init__(
        self,
        config: TrainingConfig,
        word_to_id: dict[str, int],
        id_to_word: dict[int, str],
    ) -> None:
        """Initialize model parameters and geometry-specific optimizer groups."""
        super().__init__()
        validate_config(config)
        util.PROJ_EPS = config.proj_eps
        self.config = config
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.dtype = torch.float64
        self.embeddings = _initial_embedding(len(word_to_id), config, self.dtype)
        self.eucl_params = []
        self.hyp_params = []
        if config.inputs_geom == "eucl":
            self.eucl_params.append(self.embeddings)
        self.encoder_1 = self._build_encoder()
        self.encoder_2 = self._build_encoder()
        self.eucl_params.extend(self._euclidean_encoder_params(self.encoder_1))
        self.eucl_params.extend(self._euclidean_encoder_params(self.encoder_2))
        self.hyp_params.extend(self._hyperbolic_encoder_params(self.encoder_1))
        self.hyp_params.extend(self._hyperbolic_encoder_params(self.encoder_2))
        self.W_ff_s1 = nn.Parameter(
            torch.empty(config.word_dim, config.before_mlr_dim, dtype=self.dtype)
        )
        self.W_ff_s2 = nn.Parameter(
            torch.empty(config.word_dim, config.before_mlr_dim, dtype=self.dtype)
        )
        nn.init.xavier_uniform_(self.W_ff_s1)
        nn.init.xavier_uniform_(self.W_ff_s2)
        self.b_ff = nn.Parameter(
            torch.zeros(1, config.before_mlr_dim, dtype=self.dtype)
        )
        self.b_ff_d = nn.Parameter(
            torch.zeros(1, config.before_mlr_dim, dtype=self.dtype)
        )
        self.A_mlr = nn.Parameter(
            torch.empty(config.num_classes, config.before_mlr_dim, dtype=self.dtype)
        )
        self.P_mlr = nn.Parameter(
            torch.zeros(config.num_classes, config.before_mlr_dim, dtype=self.dtype)
        )
        nn.init.xavier_uniform_(self.A_mlr)
        self._register_parameter_groups()

    def _build_encoder(self) -> nn.Module:
        """Build one sentence encoder cell for inputs with shape [batch, tokens, word_dim]."""
        config = self.config
        if config.cell_type == "TFrnn":
            return nn.RNNCell(config.word_dim, config.word_dim, dtype=self.dtype)
        if config.cell_type == "TFgru":
            return nn.GRUCell(config.word_dim, config.word_dim, dtype=self.dtype)
        if config.cell_type == "TFlstm":
            return nn.LSTMCell(config.word_dim, config.word_dim, dtype=self.dtype)
        if config.cell_type == "rnn" and config.sent_geom == "eucl":
            return rnn_impl.EuclRNN(config.word_dim, config.word_dim, dtype=self.dtype)
        if config.cell_type == "gru" and config.sent_geom == "eucl":
            return rnn_impl.EuclGRU(config.word_dim, config.word_dim, dtype=self.dtype)
        if config.cell_type == "rnn" and config.sent_geom == "hyp":
            return rnn_impl.HypRNN(
                input_dim=config.word_dim,
                num_units=config.word_dim,
                inputs_geom=config.inputs_geom,
                bias_geom=config.bias_geom,
                c_val=config.c,
                non_lin=config.cell_non_lin,
                fix_biases=config.fix_biases,
                fix_matrices=config.fix_matrices,
                matrices_init_eye=config.matrices_init_eye,
                dtype=self.dtype,
            )
        if config.cell_type == "gru" and config.sent_geom == "hyp":
            return rnn_impl.HypGRU(
                input_dim=config.word_dim,
                num_units=config.word_dim,
                inputs_geom=config.inputs_geom,
                bias_geom=config.bias_geom,
                c_val=config.c,
                non_lin=config.cell_non_lin,
                fix_biases=config.fix_biases,
                fix_matrices=config.fix_matrices,
                matrices_init_eye=config.matrices_init_eye,
                dtype=self.dtype,
            )
        raise ValueError("Invalid cell and sentence geometry combination")

    def _euclidean_encoder_params(self, encoder: nn.Module) -> list[nn.Parameter]:
        """Return Euclidean trainable parameters for one encoder."""
        if hasattr(encoder, "eucl_params"):
            return encoder.eucl_params
        return list(encoder.parameters())

    def _hyperbolic_encoder_params(self, encoder: nn.Module) -> list[nn.Parameter]:
        """Return hyperbolic trainable parameters for one encoder."""
        if hasattr(encoder, "hyp_params"):
            return encoder.hyp_params
        return []

    def _register_parameter_groups(self) -> None:
        """Collect Euclidean and hyperbolic parameter groups for separate optimizers."""
        config = self.config
        self.eucl_params.extend([self.W_ff_s1, self.W_ff_s2, self.A_mlr])
        if config.ffnn_geom == "eucl" or config.bias_geom == "eucl":
            self.eucl_params.append(self.b_ff)
            if config.additional_features == "dsq":
                self.eucl_params.append(self.b_ff_d)
        else:
            self.hyp_params.append(self.b_ff)
            if config.additional_features == "dsq":
                self.hyp_params.append(self.b_ff_d)
        if config.mlr_geom == "eucl":
            self.eucl_params.append(self.P_mlr)
        if config.mlr_geom == "hyp":
            self.hyp_params.append(self.P_mlr)

    def encode_sentence(
        self, word_ids: torch.Tensor, lengths: torch.Tensor, encoder: nn.Module
    ) -> torch.Tensor:
        """Encode padded token ids [batch, tokens] into states [batch, word_dim]."""
        word_embeddings = F.embedding(word_ids, self.embeddings)
        batch_size = word_embeddings.shape[0]
        state = torch.zeros(
            batch_size,
            self.config.word_dim,
            dtype=self.dtype,
            device=word_embeddings.device,
        )
        cell_state = torch.zeros(
            batch_size,
            self.config.word_dim,
            dtype=self.dtype,
            device=word_embeddings.device,
        )
        for token_index in range(word_embeddings.shape[1]):
            active_mask = (lengths > token_index).to(dtype=self.dtype).unsqueeze(1)
            previous_state = state
            if self.config.cell_type == "TFlstm":
                new_state, new_cell_state = encoder(
                    word_embeddings[:, token_index, :], (state, cell_state)
                )
                state = active_mask * new_state + (1.0 - active_mask) * previous_state
                cell_state = (
                    active_mask * new_cell_state + (1.0 - active_mask) * cell_state
                )
            else:
                new_state = encoder(word_embeddings[:, token_index, :], state)
                state = active_mask * new_state + (1.0 - active_mask) * previous_state
        return state

    def forward(
        self, batch: Batch, dropout_keep_prob: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits [batch, classes] and distance squared [batch, 1]."""
        config = self.config
        sent_1 = self.encode_sentence(
            batch.word_ids_1, batch.num_words_1, self.encoder_1
        )
        sent_2 = self.encode_sentence(
            batch.word_ids_2, batch.num_words_2, self.encoder_2
        )
        if config.sent_geom == "eucl":
            distance_sq = util.euclid_dist_sq(sent_1, sent_2)
        else:
            distance_sq = util.poinc_dist_sq(sent_1, sent_2, c=config.c)
        if config.ffnn_geom == "eucl" and config.sent_geom == "hyp":
            sent_1 = util.log_map_zero(sent_1, config.c)
            sent_2 = util.log_map_zero(sent_2, config.c)
        if config.ffnn_geom == "eucl":
            output_ffnn = (
                torch.matmul(sent_1, self.W_ff_s1)
                + torch.matmul(sent_2, self.W_ff_s2)
                + self.b_ff
            )
            if config.additional_features == "dsq":
                output_ffnn = output_ffnn + distance_sq * self.b_ff_d
        else:
            ffnn_s1 = util.mob_mat_mul(self.W_ff_s1, sent_1, config.c)
            ffnn_s2 = util.mob_mat_mul(self.W_ff_s2, sent_2, config.c)
            output_ffnn = util.mob_add(ffnn_s1, ffnn_s2, config.c)
            hyp_b_ff = self.b_ff
            if config.bias_geom == "eucl":
                hyp_b_ff = util.exp_map_zero(self.b_ff, config.c)
            output_ffnn = util.mob_add(output_ffnn, hyp_b_ff, config.c)
            if config.additional_features == "dsq":
                hyp_b_ff_d = self.b_ff_d
                if config.bias_geom == "eucl":
                    hyp_b_ff_d = util.exp_map_zero(self.b_ff_d, config.c)
                output_ffnn = util.mob_add(
                    output_ffnn,
                    util.mob_scalar_mul(distance_sq, hyp_b_ff_d, config.c),
                    config.c,
                )
        if config.ffnn_geom == "eucl":
            output_ffnn = util.eucl_non_lin(output_ffnn, non_lin=config.ffnn_non_lin)
        else:
            output_ffnn = util.hyp_non_lin(
                output_ffnn,
                non_lin=config.ffnn_non_lin,
                hyp_output=(config.mlr_geom == "hyp" and dropout_keep_prob == 1.0),
                c=config.c,
            )
        if dropout_keep_prob < 1.0:
            output_ffnn = F.dropout(
                output_ffnn, p=1.0 - dropout_keep_prob, training=self.training
            )
            if config.mlr_geom == "hyp":
                output_ffnn = util.exp_map_zero(output_ffnn, config.c)
        logits_list = []
        for class_index in range(config.num_classes):
            class_a = self.A_mlr[class_index : class_index + 1]
            class_p = self.P_mlr[class_index : class_index + 1]
            if config.mlr_geom == "eucl":
                logits_list.append(
                    util.dot(-class_p + output_ffnn, class_a).reshape(-1)
                )
            if config.mlr_geom == "hyp":
                minus_p_plus_x = util.mob_add(-class_p, output_ffnn, config.c)
                norm_a = util.norm(class_a)
                lambda_px = util.lambda_x(minus_p_plus_x, config.c)
                px_dot_a = util.dot(minus_p_plus_x, F.normalize(class_a, p=2, dim=1))
                logit = (
                    2.0
                    / np.sqrt(config.c)
                    * norm_a
                    * torch.asinh(np.sqrt(config.c) * px_dot_a * lambda_px)
                )
                logits_list.append(logit.reshape(-1))
        logits = torch.stack(logits_list, dim=1)
        return logits, distance_sq


def compute_loss(
    config: TrainingConfig,
    logits: torch.Tensor,
    labels: torch.Tensor,
    distance_sq: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy plus optional distance regularizer."""
    loss = F.cross_entropy(logits, labels)
    if config.reg_beta > 0.0:
        if config.num_classes != 2:
            raise ValueError("Distance regularizer expects two classes")
        distance_regularizer = torch.mean(
            (labels.to(dtype=torch.float64) - 0.5).reshape(-1, 1) * distance_sq
        )
        loss = loss + config.reg_beta * distance_regularizer
    return loss


class RiemannianTrainer:
    """Train a `HyperbolicRNNModel` with Adam and manual RSGD updates."""

    def __init__(self, config: TrainingConfig, model: HyperbolicRNNModel) -> None:
        """Initialize Adam for Euclidean params and keep hyperbolic params manual."""
        self.config = config
        self.model = model
        self.euclidean_optimizer = (
            torch.optim.Adam(model.eucl_params, lr=1e-3)
            if len(model.eucl_params) > 0
            else None
        )

    def zero_grad(self) -> None:
        """Clear gradients for Euclidean and hyperbolic parameters."""
        if self.euclidean_optimizer is not None:
            self.euclidean_optimizer.zero_grad()
        for parameter in self.model.hyp_params:
            if parameter.grad is not None:
                parameter.grad.zero_()
        if self.config.inputs_geom == "hyp" and self.model.embeddings.grad is not None:
            self.model.embeddings.grad.zero_()

    def rsgd_update(
        self,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        learning_rate: float,
        burn_in_factor: float,
    ) -> torch.Tensor:
        """Return an RSGD or projected-SGD update for hyperbolic tensors [batch, dim]."""
        riemannian_rescaling_factor = util.riemannian_gradient_c(
            parameter, c=self.config.c
        )
        clipped_gradient = self._clip_gradient_tensor(gradient)
        rescaled_gradient = riemannian_rescaling_factor * clipped_gradient
        if self.config.hyp_opt == "rsgd":
            return util.exp_map_x(
                parameter,
                -burn_in_factor * learning_rate * rescaled_gradient,
                c=self.config.c,
            )
        updated_parameter = (
            parameter - burn_in_factor * learning_rate * rescaled_gradient
        )
        return util.project_hyp_vecs(updated_parameter, self.config.c)

    def _clip_gradient_tensor(self, gradient: torch.Tensor) -> torch.Tensor:
        """Clip one gradient tensor by global norm while preserving shape."""
        gradient_norm = torch.linalg.norm(gradient).clamp_min(util.EPS)
        scale = (1.0 / gradient_norm).clamp_max(1.0)
        return gradient * scale

    def _update_hyperbolic_embeddings(
        self, batch: Batch, burn_in_factor: float
    ) -> None:
        """Aggregate duplicate word ids and update embeddings with RSGD."""
        if self.config.inputs_geom != "hyp" or self.model.embeddings.grad is None:
            return
        indices = torch.cat(
            [batch.word_ids_1.reshape(-1), batch.word_ids_2.reshape(-1)]
        ).unique()
        gradients = self.model.embeddings.grad.index_select(0, indices)
        clipped_gradients = self._clip_gradient_tensor(gradients)
        unique_embeddings = self.model.embeddings.detach().index_select(0, indices)
        riemannian_rescaling_factor = util.riemannian_gradient_c(
            unique_embeddings, c=self.config.c
        )
        rescaled_gradient = riemannian_rescaling_factor * clipped_gradients
        if self.config.hyp_opt == "rsgd":
            updated_embeddings = util.exp_map_x(
                unique_embeddings,
                -burn_in_factor * self.config.lr_words * rescaled_gradient,
                c=self.config.c,
            )
        else:
            updated_embeddings = util.project_hyp_vecs(
                unique_embeddings
                - burn_in_factor * self.config.lr_words * rescaled_gradient,
                self.config.c,
            )
        self.model.embeddings.data.index_copy_(0, indices, updated_embeddings)

    def _update_hyperbolic_parameters(self, burn_in_factor: float) -> None:
        """Update non-embedding hyperbolic parameters with RSGD/projection."""
        for parameter in self.model.hyp_params:
            if parameter.grad is not None:
                clipped_gradient = self._clip_gradient_tensor(parameter.grad)
                riemannian_rescaling_factor = util.riemannian_gradient_c(
                    parameter.detach(), c=self.config.c
                )
                rescaled_gradient = riemannian_rescaling_factor * clipped_gradient
                if self.config.hyp_opt == "rsgd":
                    updated_parameter = util.exp_map_x(
                        parameter.detach(),
                        -burn_in_factor * self.config.lr_ffnn * rescaled_gradient,
                        c=self.config.c,
                    )
                else:
                    updated_parameter = util.project_hyp_vecs(
                        parameter.detach()
                        - burn_in_factor * self.config.lr_ffnn * rescaled_gradient,
                        self.config.c,
                    )
                parameter.data.copy_(updated_parameter)

    def train_batch(self, batch: Batch, burn_in_factor: float) -> float:
        """Run one optimizer step and return scalar loss."""
        self.model.train()
        self.zero_grad()
        logits, distance_sq = self.model(batch, dropout_keep_prob=self.config.dropout)
        loss = compute_loss(self.config, logits, batch.labels, distance_sq)
        loss.backward()
        for parameter in self.model.eucl_params:
            if parameter.grad is not None:
                torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        if self.euclidean_optimizer is not None:
            self.euclidean_optimizer.step()
        with torch.no_grad():
            self._update_hyperbolic_embeddings(
                batch=batch, burn_in_factor=burn_in_factor
            )
            self._update_hyperbolic_parameters(burn_in_factor=burn_in_factor)
        return float(loss.detach().cpu().item())

    def max_hyperbolic_norm(self) -> float:
        """Return the largest norm among hyperbolic parameter tensors."""
        max_norm = 0.0
        tensors = list(self.model.hyp_params)
        if self.config.inputs_geom == "hyp":
            tensors.append(self.model.embeddings)
        for tensor in tensors:
            current_norm = (
                torch.max(torch.linalg.norm(tensor.detach(), dim=1)).cpu().item()
            )
            max_norm = max(max_norm, current_norm)
        return float(max_norm)


def evaluate(
    model: HyperbolicRNNModel,
    config: TrainingConfig,
    data: Sequence[SentencePair],
    device: torch.device,
) -> float:
    """Compute classification accuracy over a dataset."""
    model.eval()
    predictions = []
    with torch.no_grad():
        item_index = 0
        while item_index < len(data):
            if (
                config.max_eval_batches > 0
                and item_index >= config.max_eval_batches * config.batch_size
            ):
                break
            batch = next_batch(
                i=item_index, batch_size=config.batch_size, data=data
            ).to_device(device)
            logits, _ = model(batch, dropout_keep_prob=1.0)
            batch_predictions = torch.argmax(logits, dim=1).detach().cpu().tolist()
            predictions.extend(batch_predictions)
            item_index += config.batch_size
    num_scored_examples = min(len(predictions), len(data))
    predictions = predictions[:num_scored_examples]
    num_correct = 0
    for item_index, predicted_label in enumerate(predictions):
        if predicted_label == data[item_index][4]:
            num_correct += 1
    return num_correct / float(num_scored_examples)


def log_dataset_stats(logger: object, name: str, data: Sequence[SentencePair]) -> None:
    """Log split size and class percentages."""
    logger.info(name + " data size: %d" % len(data))
    class_to_count = {1: 0.0, 0: 0.0}
    for item in data:
        class_to_count[item[4]] += 1.0
    for class_label in class_to_count:
        logger.info(
            "Class %d has %.4f percent samples"
            % (class_label, 100.0 * class_to_count[class_label] / len(data))
        )


def train_model(
    config: TrainingConfig,
    model: HyperbolicRNNModel,
    training_data: Sequence[SentencePair],
    dev_data: Sequence[SentencePair],
    test_data: Sequence[SentencePair],
    logger: object,
    experiment_name: str,
) -> None:
    """Train the model and write TensorBoard/checkpoint artifacts."""
    device = resolve_device(config)
    model.to(device)
    trainer = RiemannianTrainer(config=config, model=model)
    writer = SummaryWriter(os.path.join(config.output_dir, "tb_28may", experiment_name))
    if config.restore_model:
        state_dict = torch.load(config.restore_from_path, map_location=device)
        model.load_state_dict(state_dict)
    training_data_batches = dataset_to_minibatches(training_data, config.batch_size)
    best_test_accuracy = 0.0
    best_validation_accuracy = 0.0
    best_i = 0
    burn_in_factor = 1.0
    epoch = 0
    while epoch < config.num_epochs:
        logger.info("Epoch: %d" % epoch)
        cur_total_time = 0.0
        random.shuffle(training_data_batches)
        if config.burnin and epoch == 0:
            burn_in_factor = burn_in_factor / 10.0
        batch_index = 0
        while batch_index < len(training_data_batches):
            if config.max_train_batches > 0 and batch_index >= config.max_train_batches:
                break
            batch = training_data_batches[batch_index].to_device(device)
            sess_time_start = time.perf_counter()
            current_loss = trainer.train_batch(
                batch=batch, burn_in_factor=burn_in_factor
            )
            cur_total_time += time.perf_counter() - sess_time_start
            global_step = epoch * len(training_data) + batch_index * config.batch_size
            writer.add_scalar("classif/unreg_loss", current_loss, global_step)
            if batch_index % config.print_step == 0:
                if batch_index > 0:
                    avg_sec_per_sent = cur_total_time / (
                        config.print_step * config.batch_size
                    )
                    logger.info(
                        "Num examples processed: %d. curr_loss: %.4f; sec_per_sent: %.4f"
                        % (global_step, current_loss, avg_sec_per_sent)
                    )
                cur_total_time = 0.0
                validation_accuracy = evaluate(
                    model=model, config=config, data=dev_data, device=device
                )
                test_accuracy = evaluate(
                    model=model, config=config, data=test_data, device=device
                )
                writer.add_scalar(
                    "classif/validation_accuracy", validation_accuracy, global_step
                )
                writer.add_scalar("classif/test_accuracy", test_accuracy, global_step)
                logger.info(
                    "CURRENT val accuracy: %.4f ; test accuracy: %.4f"
                    % (validation_accuracy, test_accuracy)
                )
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_test_accuracy = test_accuracy
                    best_i = global_step
                logger.info(
                    "BEST: i = %d, val acc: %.2f, test acc: %.2f"
                    % (
                        best_i,
                        100.0 * best_validation_accuracy,
                        100.0 * best_test_accuracy,
                    )
                )
                logger.info("EXPERIMENT = " + experiment_name)
                logger.info(
                    "============================================================="
                )
            if np.isinf(current_loss) or np.isnan(current_loss):
                raise FloatingPointError(
                    "Non-finite loss at example " + str(global_step)
                )
            batch_index += 1
        epoch += 1
    checkpoint_path = os.path.join(config.output_dir, "models", experiment_name + ".pt")
    torch.save(model.state_dict(), checkpoint_path)
    writer.close()
    logger.info(
        "DONE -- BEST: i = %d, val acc: %.2f, test acc: %.2f"
        % (best_i, 100.0 * best_validation_accuracy, 100.0 * best_test_accuracy)
    )


def run() -> None:
    """Run CLI training end to end."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    os.makedirs(os.path.join(config.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "tb_28may"), exist_ok=True)
    experiment_name = make_experiment_name(config)
    logger = util.setup_logger(
        experiment_name,
        logs_dir=os.path.join(config.output_dir, "logs"),
        also_stdout=True,
    )
    logger.info("PARAMS :  " + experiment_name)
    logger.info("")
    logger.info(config)
    word_to_id, id_to_word = load_vocabulary(config)
    training_data, dev_data, test_data = load_datasets(config)
    log_dataset_stats(logger=logger, name="Training", data=training_data)
    log_dataset_stats(logger=logger, name="Validation", data=dev_data)
    log_dataset_stats(logger=logger, name="Test", data=test_data)
    model = HyperbolicRNNModel(
        config=config, word_to_id=word_to_id, id_to_word=id_to_word
    )
    train_model(
        config=config,
        model=model,
        training_data=training_data,
        dev_data=dev_data,
        test_data=test_data,
        logger=logger,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    run()
