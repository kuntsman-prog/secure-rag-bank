"""Тесты векторного поиска: фильтр доступа внутри Qdrant и эшелонированная проверка."""
import pytest

from securerag.access import can_read
from securerag.evaluate import run
from securerag.pipeline import SecureRAG
from securerag.vector import EmbedderUnavailable, HashingEmbedder, OllamaEmbedder, VectorIndex, rrf


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    return SecureRAG(log_file=tmp_path_factory.mktemp("logs") / "audit.jsonl",
                     retrieval="hybrid", embedder="hashing")


def test_qdrant_filter_equals_can_read(rag):
    # Без порога и с большим k Qdrant возвращает ровно то, что разрешает can_read
    vi = VectorIndex(rag.index.chunks, HashingEmbedder(), min_score=-1.0)
    for name, user in rag.users.items():
        found = {c.chunk_id for c, _ in vi.search(user, "регламент банка", k=100)}
        allowed = {c.chunk_id for c in rag.index.chunks if can_read(user, c.classification, list(c.departments))}
        assert found == allowed, name
    assert vi.filter_violations == 0


def test_exact_text_of_forbidden_doc_not_found(rag):
    secret = next(c for c in rag.index.chunks if c.doc_id == "DOC-003" and "15 минут" in c.text)
    hits = rag.vector.search(rag.user("ivanov"), secret.text, k=10)
    assert all(c.doc_id not in {"DOC-003", "DOC-004"} for c, _ in hits)


def test_defense_in_depth_when_filter_is_broken(rag, monkeypatch):
    # Имитация ошибки: фильтр не передан в Qdrant. Повторная проверка в коде не пропускает закрытое.
    vi = VectorIndex(rag.index.chunks, HashingEmbedder(), min_score=-1.0)
    monkeypatch.setattr(vi, "_filter", lambda user: None)
    hits = vi.search(rag.user("ivanov"), "время реакции SOC на критический инцидент", k=100)
    assert hits and all(c.doc_id not in {"DOC-003", "DOC-004"} for c, _ in hits)
    assert vi.filter_violations > 0


def test_hybrid_eval_no_leaks(rag):
    report = run(rag)
    assert report["retrieval"] == "hybrid:hashing"
    assert report["leaks"] == 0 and report["failures"] == []
    assert report["vector_filter_violations"] == 0


def test_rrf_rewards_agreement(rag):
    a, b, c = rag.index.chunks[:3]
    fused = [x.chunk_id for x, _ in rrf([[a, b], [b, c]])]
    assert fused[0] == b.chunk_id  # найден обоими методами


def test_unavailable_embedder_fails_loudly():
    with pytest.raises(EmbedderUnavailable):
        OllamaEmbedder(url="http://127.0.0.1:9").embed(["тест"])
