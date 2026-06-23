"""Tests for hyp_rnn YAML config loading."""

import os
import sys

DEFAULT_SERVICE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "services", "hyperbolic-nn")
)
SERVICE_DIR = os.environ.get("HYPERBOLIC_NN_SERVICE_DIR", DEFAULT_SERVICE_DIR)
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)

import hyp_rnn


def test_load_smoke_yaml_defaults() -> None:
    """Smoke config YAML loads the expected training defaults."""
    config_path = os.path.join(SERVICE_DIR, "configs", "smoke.yaml")
    defaults = hyp_rnn.load_yaml_training_defaults(config_path)

    assert defaults["base_name"] == "smoke"
    assert defaults["dataset"] == "PRFX10"
    assert defaults["batch_size"] == 2
    assert defaults["max_train_batches"] == 1
    assert defaults["burnin"] is False


def test_cli_overrides_yaml_config() -> None:
    """Explicit CLI flags override values from the YAML config file."""
    config_path = os.path.join(SERVICE_DIR, "configs", "smoke.yaml")
    config = hyp_rnn.parse_training_args(
        ["--config=" + config_path, "--batch_size=7", "--device=cpu"]
    )

    assert config.batch_size == 7
    assert config.device == "cpu"
    assert config.dataset == "PRFX10"


def test_unknown_yaml_key_raises() -> None:
    """Unknown YAML keys fail fast instead of being ignored."""
    config_path = os.path.join(SERVICE_DIR, "configs", "smoke.yaml")
    invalid_path = os.path.join(os.path.dirname(config_path), "invalid_smoke_test.yaml")
    with open(config_path, encoding="utf-8") as source_file:
        contents = source_file.read()
    with open(invalid_path, "w", encoding="utf-8") as invalid_file:
        invalid_file.write(contents + "\nnot_a_real_field: true\n")

    raised = False
    try:
        hyp_rnn.load_yaml_training_defaults(invalid_path)
    except ValueError as error:
        raised = True
        assert "Unknown config key" in str(error)
    os.remove(invalid_path)

    assert raised
