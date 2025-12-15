from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders.fastembedder import FastEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore

vectorstore = QdrantVectorstore(host="localhost", port=6333)
embedder = FastEmbedder(
    model_name="Qdrant/bm25",
    embedding_name="bm25_embeddings",
)
vectorstore.create_collection("my_documents",vector_config=[VectorConfig(name="embedding", dimensions=1536)])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=1024),
        embedder,
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

pipeline.run("./docs/ricettario1.pdf")

results = vectorstore.search(query_vector = [0.0] * 1536, collection_name="my_documents", k=5)
print(results)