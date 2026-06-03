"""Tests for LocalReranker — requires sentence-transformers, marked slow."""

import pytest

from zotero_arxiv_daily.reranker.local import LocalReranker


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_local_reranker_falls_back_when_model_load_fails(config, monkeypatch):
    reranker = LocalReranker(config)

    def fail_load_encoder(self):
        raise OSError("offline")

    monkeypatch.setattr(LocalReranker, "_load_encoder", fail_load_encoder)

    score = reranker.get_similarity_score(["hello world"], ["hello there"])

    assert score.shape == (1, 1)
    assert score[0, 0] > 0
