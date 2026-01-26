# retrieval.py

import time
from qdrant_client import models, QdrantClient
from src.retrieval.chunk_embed import chunk_markdown, EmbedData, save_embeddings, load_embeddings
from src.retrieval.index import QdrantVDB

#TODO
#TESTARE convert_pdf_to_markdown() CON IL PDF

name = 'ricettario1'
query = 'cosa posso cucinare con porri e pomodori confit?'
print(query)
# se avevo già calcolato l'embeddings lo ricarico invece di ricalcolarmelo
embeddata = load_embeddings(f"embeddings_{name}.pkl")

# vector_db = QdrantVDB(collection_name=f"collection_{name}", vector_dim=len(embeddata.embeddings[0]), batch_size=7)
vector_db = QdrantClient(host="localhost", port=6333)
query_embedding = embeddata.embed_model.get_query_embedding(query)
top_k=7

result = vector_db.search(
    collection_name=f"collection_{name}",
    query_vector=query_embedding,
    limit=top_k,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=True,
            rescore=True,   # re-ranking with vector similarity
            oversampling=2.0,
        )
    ),
    timeout=1000,
)


print(result)
