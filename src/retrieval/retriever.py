# retrieval.py

import time
from qdrant_client import models

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query, top_k=7):
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)

        start_time = time.time()
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
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
        end_time = time.time()
        print(f"Execution time for the search: {end_time - start_time:.4f} seconds")

        return result
