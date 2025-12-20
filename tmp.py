import os
from datapizza.clients.openai import OpenAIClient
from datapizza.core.vectorstore import VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.clients.openai.openai_client import OpenAIClient
from datapizza.modules.rewriters import ToolRewriter
from dotenv import load_dotenv


load_dotenv(override=True)



vector_store = QdrantVectorstore(host="localhost", port=6333)

vector_store.create_collection(collection_name="datapizza", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = IngestionPipeline(
    modules=[
        DoclingParser(),
        NodeSplitter(max_char=2000),
        ChunkEmbedder(client=OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"), model_name="text-embedding-3-small", embedding_name="small"),
    ],
    vector_store=vector_store,
    collection_name="datapizza",
)

pipeline.run("./docs/ricettario1.pdf")

results = vector_store.search(
    query_text="ricette italiane con pasta",
    collection_name="datapizza",
    k=5
)
print(results)

# print(vector_store.search(query_vector= [0.0]*1536, collection_name="datapizza", k=4))
