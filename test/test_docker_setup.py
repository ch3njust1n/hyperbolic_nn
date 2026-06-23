"""Tests for dockerized hyperbolic-nn service layout."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SERVICE_DIR = os.path.join(REPO_ROOT, "src", "services", "hyperbolic-nn")
SERVICE_DIR = os.environ.get("HYPERBOLIC_NN_SERVICE_DIR", DEFAULT_SERVICE_DIR)
COMPOSE_PATH = os.environ.get(
    "HYPERBOLIC_NN_COMPOSE_PATH", os.path.join(REPO_ROOT, "docker-compose.yaml")
)


def test_service_layout_exists() -> None:
    """Required service files and datasets are present."""
    required_paths = [
        COMPOSE_PATH,
        os.path.join(SERVICE_DIR, "Dockerfile"),
        os.path.join(SERVICE_DIR, "hyp_rnn.py"),
        os.path.join(SERVICE_DIR, "prefix_10_dataset", "train"),
    ]
    for path in required_paths:
        assert os.path.exists(path), path


def test_output_dir_env_documented() -> None:
    """Compose file mounts OUTPUT_DIR for job artifacts."""
    with open(COMPOSE_PATH, "r", encoding="utf-8") as compose_file:
        compose_text = compose_file.read()
    assert "OUTPUT_DIR" in compose_text
    assert "/jobs" in compose_text
