"""Shared fixtures for arxiv-radar-mcp tests."""
import json
from pathlib import Path

import pytest

from arxiv_radar_mcp.config import Config, EmbeddingsConfig, ServerConfig, SourceConfig


@pytest.fixture
def sample_papers_dir(tmp_path: Path) -> Path:
    """Synthetic 'data/papers-*.json' folder with three papers across two months."""
    data = tmp_path / "data"
    data.mkdir()
    base = {
        "first_author": "Alice", "authors": ["Alice", "Bob"],
        "primary_category": "cond-mat.mtrl-sci",
        "categories": ["cond-mat.mtrl-sci"],
        "comment": None,
        "pdf_url": "http://arxiv.org/pdf/x",
        "topics": ["Quantum Chemistry & Force Fields"],
    }
    (data / "papers-2025-03.json").write_text(json.dumps({
        "2503.00001": {**base,
                       "title": "DFT study of mackinawite",
                       "abstract": "We compute lattice parameters of FeS via density functional theory.",
                       "published": "2025-03-01", "updated": "2025-03-15",
                       "tags": ["dft", "ab-initio"]},
    }), encoding="utf-8")
    (data / "papers-2025-04.json").write_text(json.dumps({
        "2504.00001": {**base,
                       "title": "MLIP for organic crystals",
                       "abstract": "Equivariant graph neural network potential trained on QM9.",
                       "published": "2025-04-15", "updated": "2025-04-15",
                       "tags": ["mlip", "gnn"]},
        "2504.00002": {**base,
                       "title": "Generative model for catalysts",
                       "abstract": "Diffusion model proposes new transition-metal complexes.",
                       "published": "2025-04-22", "updated": "2025-04-22",
                       "tags": ["generative-model", "catalysis"]},
    }), encoding="utf-8")
    return tmp_path


@pytest.fixture
def local_config(sample_papers_dir: Path, tmp_path: Path) -> Config:
    """Config pointing the loader at the synthetic corpus only (no GitHub fetch)."""
    return Config(
        sources=[SourceConfig(name="ai4chem", type="local", path=sample_papers_dir)],
        embeddings=EmbeddingsConfig(cache_dir=tmp_path / "cache"),
        server=ServerConfig(),
    )
