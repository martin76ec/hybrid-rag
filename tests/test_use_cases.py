"""Smoke tests for use cases with mock implementations."""

from hybrid_rag.application.ingest import IngestDocumentUseCase
from hybrid_rag.application.query import QueryKnowledgeBaseUseCase
from hybrid_rag.domain.ports import (
    DocumentAnalyzer,
    DocumentReader,
    EmbeddingProvider,
    VectorStore,
    LanguageModel,
    GraphStore,
    TripleExtractor,
    TripleRefiner,
)
from hybrid_rag.domain.entities import Document
from hybrid_rag.domain.value_objects import (
    DocumentAnalysis,
    EmbeddingVector,
    ChunkMetadata,
    RetrievalResult,
    Triple,
)


class FakeReader(DocumentReader):
    def read(self, source):
        return Document(
            source=source, text="Hello world this is a test document with some words."
        )


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


class FakeGraphStore(GraphStore):
    def __init__(self):
        self._triples: list[Triple] = []

    def add_triples(self, triples):
        self._triples.extend(triples)

    def query(self, question, top_k=10):
        return []

    def node_count(self):
        entities = set()
        for t in self._triples:
            entities.add(t.subject)
            entities.add(t.obj)
        return len(entities)

    def edge_count(self):
        return len(self._triples)

    def all_triples(self):
        return self._triples


class FakeTripleExtractor(TripleExtractor):
    def extract(self, text, source=""):
        return [
            Triple(
                subject="Test",
                predicate="is_a",
                obj="Document",
                source=source,
                chunk_index=0,
                chunk_text=text,
            )
        ]


class FakeTripleRefiner(TripleRefiner):
    def refine(self, raw_triples):
        return {
            "canonical_mapping": {},
            "shortened_predicates": [],
            "removed_triples": [],
            "added_triples": [],
        }


class FakeDocumentAnalyzer(DocumentAnalyzer):
    def analyze(self, text):
        return DocumentAnalysis(
            doc_type="test document",
            doc_description="A test document for unit testing.",
            suggested_triple_patterns=["entity → relates_to → concept"],
        )


def test_ingest():
    store = FakeStore()
    use_case = IngestDocumentUseCase(
        reader=FakeReader(), embedder=FakeEmbedder(), store=store
    )
    result = use_case.execute("test.pdf", chunk_size=10, overlap=2)
    assert result.source == "test.pdf"
    assert result.num_chunks > 0
    assert len(store._data) == result.num_chunks
    assert result.num_triples == 0


def test_ingest_with_graph():
    store = FakeStore()
    graph_store = FakeGraphStore()
    extractor = FakeTripleExtractor()
    refiner = FakeTripleRefiner()
    use_case = IngestDocumentUseCase(
        reader=FakeReader(),
        embedder=FakeEmbedder(),
        store=store,
        triple_extractor=extractor,
        graph_store=graph_store,
        triple_refiner=refiner,
    )
    result = use_case.execute("test.pdf", chunk_size=10, overlap=2)
    assert result.num_triples > 0
    assert graph_store.edge_count() == result.num_triples


def test_ingest_with_document_analyzer():
    store = FakeStore()
    graph_store = FakeGraphStore()
    extractor = FakeTripleExtractor()
    refiner = FakeTripleRefiner()
    doc_analyzer = FakeDocumentAnalyzer()
    use_case = IngestDocumentUseCase(
        reader=FakeReader(),
        embedder=FakeEmbedder(),
        store=store,
        triple_extractor=extractor,
        graph_store=graph_store,
        triple_refiner=refiner,
        document_analyzer=doc_analyzer,
    )
    result = use_case.execute("test.pdf", chunk_size=10, overlap=2)
    assert result.num_triples > 0
    assert result.extraction_summary is not None
    assert result.extraction_summary.doc_type == "test document"
    assert len(result.extraction_summary.suggested_triple_patterns) == 1


def test_query():
    store = FakeStore()
    uc = IngestDocumentUseCase(
        reader=FakeReader(), embedder=FakeEmbedder(), store=store
    )
    uc.execute("test.pdf", chunk_size=10, overlap=2)

    q_use_case = QueryKnowledgeBaseUseCase(
        embedder=FakeEmbedder(), store=store, llm=FakeLLM()
    )
    result = q_use_case.execute("What is this?")
    assert result.answer == "This is a fake answer."
    assert len(result.sources) > 0


def test_query_with_graph():
    store = FakeStore()
    uc = IngestDocumentUseCase(
        reader=FakeReader(), embedder=FakeEmbedder(), store=store
    )
    uc.execute("test.pdf", chunk_size=10, overlap=2)

    graph_store = FakeGraphStore()
    q_use_case = QueryKnowledgeBaseUseCase(
        embedder=FakeEmbedder(),
        store=store,
        llm=FakeLLM(),
        graph_store=graph_store,
    )
    result = q_use_case.execute("What is this?")
    assert result.answer == "This is a fake answer."
    assert len(result.sources) > 0
