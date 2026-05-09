"""Tests for the prefix registry and Encoder wiring (no model load).

Encoder._ensure_loaded is monkey-patched away; we only verify the
correct prefix is applied and the right call is dispatched. Loading the
real SentenceTransformer is integration territory — covered by
--build-cache against the live corpus.
"""
from __future__ import annotations

import numpy as np
import pytest

from corpus_core.embeddings import (Encoder, _maybe_truncate, passage_prefix, query_prefix)


# ----- prefix registry ------------------------------------------------------

@pytest.mark.parametrize("model,expected", [
    ("mixedbread-ai/mxbai-embed-large-v1",
     "Represent this sentence for searching relevant passages: "),
    ("BAAI/bge-large-en-v1.5",
     "Represent this sentence for searching relevant passages: "),
    ("BAAI/bge-small-en-v1.5",
     "Represent this sentence for searching relevant passages: "),
    ("intfloat/e5-large-v2", "query: "),
    ("intfloat/e5-base-v2", "query: "),
    ("sentence-transformers/all-MiniLM-L6-v2", ""),  # unknown → no-op
    ("totally/made-up-model", ""),
])
def test_query_prefix_registry(model: str, expected: str) -> None:
    assert query_prefix(model) == expected


@pytest.mark.parametrize("model,expected", [
    ("intfloat/e5-large-v2", "passage: "),
    ("intfloat/e5-small-v2", "passage: "),
    ("mixedbread-ai/mxbai-embed-large-v1", ""),  # mxbai is query-only
    ("BAAI/bge-large-en-v1.5", ""),              # BGE is query-only
])
def test_passage_prefix_registry(model: str, expected: str) -> None:
    assert passage_prefix(model) == expected


# ----- Encoder applies prefixes correctly ------------------------------------

class _FakeSentenceTransformer:
    """Captures what the real SentenceTransformer would have been asked to encode."""

    def __init__(self) -> None:
        self.last_call: list[str] | None = None

    def encode(self, texts, **kwargs) -> np.ndarray:
        self.last_call = list(texts)
        # Return whatever shape the caller expects (single or batch).
        return np.ones((len(texts), 4), dtype=np.float32)


def _patch_encoder(enc: Encoder, fake: _FakeSentenceTransformer) -> None:
    """Replace the lazy load with a stub that returns our fake model."""
    enc._model = fake


def test_encoder_query_applies_mxbai_prefix(local_config):
    local_config.embeddings.model = "mixedbread-ai/mxbai-embed-large-v1"
    enc = Encoder(local_config)
    fake = _FakeSentenceTransformer()
    _patch_encoder(enc, fake)

    enc.encode_query("what is dft?")

    assert fake.last_call == [
        "Represent this sentence for searching relevant passages: what is dft?"
    ]


def test_encoder_passages_no_prefix_for_mxbai(local_config):
    local_config.embeddings.model = "mixedbread-ai/mxbai-embed-large-v1"
    enc = Encoder(local_config)
    fake = _FakeSentenceTransformer()
    _patch_encoder(enc, fake)

    enc.encode_passages(["a paper title", "another title"], show_progress=False)

    assert fake.last_call == ["a paper title", "another title"]


def test_encoder_passages_apply_e5_prefix(local_config):
    local_config.embeddings.model = "intfloat/e5-base-v2"
    enc = Encoder(local_config)
    fake = _FakeSentenceTransformer()
    _patch_encoder(enc, fake)

    enc.encode_passages(["doc one", "doc two"], show_progress=False)

    assert fake.last_call == ["passage: doc one", "passage: doc two"]


def test_encoder_query_no_prefix_for_unknown_model(local_config):
    local_config.embeddings.model = "totally/made-up-model"
    enc = Encoder(local_config)
    fake = _FakeSentenceTransformer()
    _patch_encoder(enc, fake)

    enc.encode_query("hello world")

    assert fake.last_call == ["hello world"]


def test_encoder_query_uses_qwen3_instruct_template(local_config):
    local_config.embeddings.model = "Qwen/Qwen3-Embedding-8B"
    enc = Encoder(local_config)
    fake = _FakeSentenceTransformer()
    _patch_encoder(enc, fake)

    enc.encode_query("MLIP for organic crystals")

    assert fake.last_call == [
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: MLIP for organic crystals"
    ]


# ----- matryoshka truncation ------------------------------------------------

def test_maybe_truncate_noop_when_target_dim_none() -> None:
    v = np.ones(8, dtype=np.float32) / np.sqrt(8)
    out = _maybe_truncate(v, None)
    assert np.array_equal(out, v)


def test_maybe_truncate_slices_and_renormalizes() -> None:
    # Pre-normalized 8-dim vector; truncating to 4 dim should give a
    # length-1 vector spanning the first 4 components.
    v = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    v = v / np.linalg.norm(v)
    out = _maybe_truncate(v, target_dim=4)
    assert out.shape == (4,)
    assert np.allclose(np.linalg.norm(out), 1.0, atol=1e-5)
    # All four kept components were equal → still equal after re-normalize.
    assert np.allclose(out, np.full(4, 0.5, dtype=np.float32), atol=1e-5)


def test_maybe_truncate_handles_2d_batch() -> None:
    batch = np.eye(3, 8, dtype=np.float32)  # rows: e_0, e_1, e_2 in 8-dim
    out = _maybe_truncate(batch, target_dim=4)
    assert out.shape == (3, 4)
    # Each row was a unit basis vector → after slicing & re-norm, still unit.
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-5)


def test_maybe_truncate_noop_when_target_ge_native() -> None:
    v = np.ones(4, dtype=np.float32) / 2
    out = _maybe_truncate(v, target_dim=8)
    assert np.array_equal(out, v)
