# retrieval.py

import time
import logging
from qdrant_client import models

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata
        logger.info("retriever initialized collection=%s", self.vector_db.collection_name)

    def search(self, query, top_k=7):
        logger.info("vector search start collection=%s top_k=%d query_length=%d", self.vector_db.collection_name, top_k, len(query))
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
        logger.info(
            "vector search complete collection=%s hits=%d duration=%.4fs",
            self.vector_db.collection_name,
            len(result),
            end_time - start_time,
        )

        return result
