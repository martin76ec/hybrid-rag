"""Smoke tests for use cases with mock implementations."""
from hybrid_rag.application.ingest import IngestDocumentUseCase
from hybrid_rag.application.query import QueryKnowledgeBaseUseCase
from hybrid_rag.domain.ports import DocumentReader, EmbeddingProvider, VectorStore, LanguageModel
from hybrid_rag.domain.entities import Document
from hybrid_rag.domain.value_objects import EmbeddingVector, ChunkMetadata, RetrievalResult


class FakeReader(DocumentReader):
    def read(self, source):
        return Document(source=source, text="Hello world this is a test document with some words.")


class FakeEmbedder(EmbeddingProvider):
    def embed(self, text):
        return EmbeddingVector(values=[1.0, 0.0, 0.0])

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class FakeStore(VectorStore):
    def __init__(self):
        self._data = []

    def add(self, vectors, metadata):
        for v, m in zip(vectors, metadata):
            self._data.append((v, m))

    def search(self, query, top_k):
        return [
            RetrievalResult(chunk=self._data[i][1], score=0.0)
            for i in range(min(top_k, len(self._data)))
        ]


class FakeLLM(LanguageModel):
    def generate(self, prompt):
        return "This is a fake answer."


def test_ingest():
    store = FakeStore()
    use_case = IngestDocumentUseCase(reader=FakeReader(), embedder=FakeEmbedder(), store=store)
    result = use_case.execute("test.pdf", chunk_size=10, overlap=2)
    assert result.source == "test.pdf"
    assert result.num_chunks > 0
    assert len(store._data) == result.num_chunks


def test_query():
    store = FakeStore()
    uc = IngestDocumentUseCase(reader=FakeReader(), embedder=FakeEmbedder(), store=store)
    uc.execute("test.pdf", chunk_size=10, overlap=2)

    q_use_case = QueryKnowledgeBaseUseCase(
        embedder=FakeEmbedder(), store=store, llm=FakeLLM()
    )
    result = q_use_case.execute("What is this?")
    assert result.answer == "This is a fake answer."
    assert len(result.sources) > 0