"""Phase 3 tests: embeddings, FAISS vector store, retrieval, evaluation.

Core integration tests use the real local embedding model and real FAISS —
nothing is mocked. The model and index are built once per session (fixtures
below) to keep the suite fast.
"""

import numpy as np
import pytest

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.evaluation import (
    EVALUATION_SET,
    evaluate_retrieval,
)
from app.rag.models import DocumentSource, DocumentType, KnowledgeDocument
from app.rag.retriever import KnowledgeRetriever
from app.rag.vector_store import (
    FaissVectorStore,
    StaleIndexError,
    embedding_text_for,
    load_knowledge_documents,
    sha256_of_file,
)


# ---------------------------------------------------------------------------
# 1-2. Model loads; dimension valid
# ---------------------------------------------------------------------------
def test_embedding_model_loads(embedder):
    assert embedder.model_name == get_settings().embedding_model
    assert "all-MiniLM-L6-v2" in embedder.model_name


def test_embedding_dimension_discovered_and_valid(embedder):
    dim = embedder.dimension
    assert isinstance(dim, int) and dim > 0
    # Cross-check against the raw model API (name varies across ST versions).
    getter = getattr(embedder._model, "get_embedding_dimension", None)
    if getter is None:
        getter = embedder._model.get_sentence_embedding_dimension
    assert dim == getter()
    # all-MiniLM-L6-v2 is documented as 384-dimensional; assert consistency,
    # not a hardcoded constant used by the code.
    probe = embedder.embed_document("dimension probe")
    assert probe.shape == (dim,)
    assert dim == 384


# ---------------------------------------------------------------------------
# 3-5. Single/batch embedding, normalization
# ---------------------------------------------------------------------------
def test_single_document_embedding(embedder):
    vector = embedder.embed_document("The marks table stores exam results.")
    assert vector.shape == (embedder.dimension,)
    assert vector.dtype == np.float32


def test_batch_embedding_shapes_and_consistency(embedder):
    texts = ["students table", "attendance percentage", "foreign key"]
    batch = embedder.embed_documents(texts)
    single = embedder.embed_documents([texts[0]])[0]
    assert batch.shape == (3, embedder.dimension)
    np.testing.assert_allclose(batch[0], single, atol=1e-5)


def test_embeddings_are_normalized(embedder):
    vectors = embedder.embed_documents(["a", "b", "c"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 6. All 38 documents embed
# ---------------------------------------------------------------------------
def test_all_knowledge_documents_embed(rag_documents, embedder):
    vectors = embedder.embed_documents(
        [embedding_text_for(doc) for doc in rag_documents]
    )
    assert vectors.shape == (len(rag_documents), 384)
    assert len(rag_documents) == 38


# ---------------------------------------------------------------------------
# 7-11. FAISS build / save / load / mapping
# ---------------------------------------------------------------------------
def test_index_created_with_correct_vector_count(rag_store, rag_documents):
    assert rag_store.size == len(rag_documents) == 38
    assert rag_store.manifest.faiss_index_type == "IndexFlatIP"


def test_save_then_load_round_trip(tmp_path, rag_documents, embedder):
    embeddings = embedder.embed_documents(["x", "y"])
    docs = rag_documents[:2]
    original = FaissVectorStore.build(
        documents=docs, embeddings=embeddings,
        embedding_model=embedder.model_name, knowledge_sha256="a" * 64,
    )
    directory = tmp_path / "roundtrip"
    original.save(directory)

    loaded = FaissVectorStore.load(directory, expected_knowledge_sha256="a" * 64,
                                   expected_embedding_model=embedder.model_name)
    assert loaded.size == original.size
    assert [d.document_id for d in loaded.documents] == [
        d.document_id for d in original.documents
    ]
    assert loaded.manifest.embedding_model == embedder.model_name
    assert loaded.manifest.knowledge_sha256 == "a" * 64


def test_document_mapping_preserved_by_position(rag_store, rag_documents):
    assert [d.document_id for d in rag_store.documents] == [
        d.document_id for d in sorted(rag_documents, key=lambda x: x.document_id)
    ]
    position_ids = {i: doc.document_id for i, doc in enumerate(rag_store.documents)}
    assert position_ids[0] == "business_rule_attendance_threshold"
    assert "schema_students" in set(position_ids.values())


def test_build_rejects_duplicate_document_ids(embedder):
    doc = KnowledgeDocument(
        document_id="dup", document_type=DocumentType.SCHEMA, title="t",
        content="c", tables=("students",),
        source=DocumentSource.POSTGRESQL_METADATA,
        extra={"table": "students"},
    )
    vectors = embedder.embed_documents(["a", "b"])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        FaissVectorStore.build(
            documents=(doc, doc.model_copy()), embeddings=vectors,
            embedding_model=embedder.model_name,
        )


def test_build_rejects_unnormalized_embeddings(rag_documents, embedder):
    bad = embedder.embed_documents(
        [embedding_text_for(d) for d in rag_documents]
    ) * 5.0  # deliberately unnormalised
    with pytest.raises(ValueError, match="[Nn]ormalis"):
        FaissVectorStore.build(
            documents=rag_documents, embeddings=bad,
            embedding_model=embedder.model_name,
        )


# ---------------------------------------------------------------------------
# 20. Stale/changed knowledge-base detection
# ---------------------------------------------------------------------------
def test_stale_digest_is_detected(rag_store_dir):
    wrong_digest = "0" * 64
    with pytest.raises(StaleIndexError, match="[Rr]ebuild"):
        FaissVectorStore.load(rag_store_dir, expected_knowledge_sha256=wrong_digest)


def test_changed_embedding_model_is_detected(rag_store_dir):
    with pytest.raises(StaleIndexError, match="[Rr]ebuild"):
        FaissVectorStore.load(
            rag_store_dir, expected_embedding_model="some-other-model"
        )


def test_missing_index_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_vector_index"):
        FaissVectorStore.load(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# 12-19. Retrieval behaviour
# ---------------------------------------------------------------------------
def test_basic_semantic_retrieval(rag_retriever):
    results = rag_retriever.retrieve(
        "Which students have attendance below 75%?", top_k=5
    )
    assert results
    top_tables = {table for r in results[:3] for table in r.tables}
    assert {"students", "attendance"} <= top_tables or any(
        r.document_id.startswith(("query_example_students_low_attendance",
                                  "business_rule_attendance"))
        for r in results[:5]
    )


def test_results_contain_full_metadata(rag_retriever):
    result = rag_retriever.retrieve("students email information", top_k=1)[0]
    payload = result.model_dump()
    assert {"document_id", "score", "document_type", "source", "tables",
            "content"} <= set(payload.keys())
    assert -1.0 <= result.score <= 1.0
    assert result.content


def test_results_are_ranked_descending(rag_retriever):
    results = rag_retriever.retrieve("department average marks", top_k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_controls_result_count(rag_retriever):
    for k in (1, 3, 10):
        results = rag_retriever.retrieve("marks constraints", top_k=k)
        assert len(results) == min(k, 38)


def test_top_k_greater_than_document_count(rag_retriever):
    results = rag_retriever.retrieve("marks", top_k=500)
    assert len(results) == 38


def test_invalid_top_k_handled_gracefully(rag_retriever):
    assert rag_retriever.retrieve("marks", top_k=0) == []
    assert rag_retriever.retrieve("marks", top_k=-3) == []


def test_empty_query_handled_gracefully(rag_retriever):
    assert rag_retriever.retrieve("", top_k=5) == []
    assert rag_retriever.retrieve("   ", top_k=5) == []


def test_no_duplicate_documents_in_results(rag_retriever):
    for question in ("marks", "students departments relationship", "attendance rule"):
        ids = [r.document_id for r in rag_retriever.retrieve(question, top_k=38)]
        assert len(ids) == len(set(ids))


def test_optional_document_type_filter(rag_retriever):
    results = rag_retriever.retrieve("allowed mark values range", top_k=5,
                                 document_type="constraint")
    assert results
    assert all(r.document_type == "constraint" for r in results)
    unfiltered = rag_retriever.retrieve("allowed mark values range", top_k=5)
    assert any(r.document_type != "constraint" for r in unfiltered)


def test_deterministic_ordering_across_instances(rag_store_dir, embedder):
    r1 = KnowledgeRetriever(embedding_service=embedder,
                            vector_store=FaissVectorStore.load(rag_store_dir))
    r2 = KnowledgeRetriever(embedding_service=embedder,
                            vector_store=FaissVectorStore.load(rag_store_dir))
    q = "Which department has the highest average marks?"
    ids1 = [r.document_id for r in r1.retrieve(q, top_k=10)]
    ids2 = [r.document_id for r in r2.retrieve(q, top_k=10)]
    assert ids1 == ids2


# ---------------------------------------------------------------------------
# Store loading helpers
# ---------------------------------------------------------------------------
def test_load_knowledge_documents_structural_validation(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        '{"document_id":"a","document_type":"schema","source":'
        '"postgresql_metadata","title":"t","content":"body","tables":[],"extra":{}}\n',
        encoding="utf-8",
    )
    docs = load_knowledge_documents(good)
    assert [d.document_id for d in docs] == ["a"]

    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"document_id":"a","document_type":"schema","source":'
        '"postgresql_metadata","title":"t","content":"body","tables":[],"extra":{}}\n'
        '{"document_id":"a","document_type":"schema","source":'
        '"postgresql_metadata","title":"t","content":"body","tables":[],"extra":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_knowledge_documents(dup)


def test_canonical_artifacts_are_current(rag_documents):
    """The checked-in rag/index must match the current knowledge.jsonl."""
    settings = get_settings()
    knowledge_path = settings.rag_output_dir / "documents" / "knowledge.jsonl"
    current_digest = sha256_of_file(knowledge_path)
    # Loading with the strict expectation raises if the committed index is stale.
    loaded = FaissVectorStore.load(
        settings.rag_index_dir,
        expected_knowledge_sha256=current_digest,
        expected_embedding_model=get_settings().embedding_model,
    )
    assert loaded.size == len(rag_documents)


def test_cached_embedding_service_is_singleton():
    assert get_embedding_service() is get_embedding_service()


# ---------------------------------------------------------------------------
# 22. Evaluation runs successfully
# ---------------------------------------------------------------------------
def test_evaluation_runs_successfully(rag_retriever):
    report = evaluate_retrieval(rag_retriever, ks=(1, 3, 5))
    assert report.total_questions == len(EVALUATION_SET) == 10
    summary = report.summary()
    assert summary["questions"] == 10
    # Sanity: recall can only improve (or stay equal) as K grows.
    assert summary["recall@1"] <= summary["recall@3"] + 1e-9
    assert summary["recall@3"] <= summary["recall@5"] + 1e-9
