"""Tests for the static docs HTTP server compose service."""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPOSE_PATH = os.environ.get(
    "HYPERBOLIC_NN_COMPOSE_PATH", os.path.join(REPO_ROOT, "docker-compose.yaml")
)
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
ENV_EXAMPLE_PATH = os.path.join(REPO_ROOT, ".env.example")

INDEX_HTML_PATH = os.path.join(DOCS_DIR, "index.html")


def test_compose_defines_docs_server() -> None:
    """Compose exposes a read-only nginx service for ./docs."""
    with open(COMPOSE_PATH, "r", encoding="utf-8") as compose_file:
        compose_text = compose_file.read()
    assert "docs-server:" in compose_text
    assert "nginx:" in compose_text
    assert "./docs:/usr/share/nginx/html:ro" in compose_text
    assert "DOCS_PORT" in compose_text


def test_docs_index_html_exists() -> None:
    """The docs server has a single index page to serve."""
    assert os.path.isfile(INDEX_HTML_PATH), INDEX_HTML_PATH


def test_docs_port_documented_in_env_example() -> None:
    """Default docs port is documented for local viewing."""
    with open(ENV_EXAMPLE_PATH, "r", encoding="utf-8") as env_example_file:
        env_example_text = env_example_file.read()
    assert "DOCS_PORT=8080" in env_example_text
