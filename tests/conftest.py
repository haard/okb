"""Shared pytest fixtures for OKB tests."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_markdown(fixtures_dir: Path) -> str:
    """Return contents of sample markdown fixture."""
    return (fixtures_dir / "sample.md").read_text()


@pytest.fixture
def sample_org(fixtures_dir: Path) -> str:
    """Return contents of sample org fixture."""
    return (fixtures_dir / "sample.org").read_text()


@pytest.fixture
def tmp_file(tmp_path: Path):
    """Factory fixture for creating temporary files with content."""

    def _create(name: str, content: str) -> Path:
        path = tmp_path / name
        path.write_text(content)
        return path

    return _create
