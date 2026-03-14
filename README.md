# Multimodal-RAG
Exam Trainer Agent lets you query your university notes in PDF format including text, tables and formulas using a Retrieval-Augmented Generation pipeline. It leverages Docling for structured PDF parsing and Qdrant for fast vector search over embedded document chunks. 



## What This Project Does
Link to Medium article reference: [Docling-Powered RAG: Querying Over Complex PDFs](https://medium.com/@pritigupta.ds/docling-powered-rag-querying-over-complex-pdfs-d99f5f58bc33)

This project is a **Streamlit-based application** on multimodal RAG that lets users:

* Upload a PDF document
* Extract structured **markdown using [Docling](https://github.com/microsoft/docling)**
* Replace embedded images with summaries
* Split the content into fixed length chunks
* Generate embeddings using `nomic-embed-text-v1.5`
* Store and index embeddings into a Qdrant vector database
* Use `GPT-5` provided by Azure AI Foundry
* Enable chat-based querying through a RAG pipeline

All of this runs **locally**, with a clean UI and persistent chat history.

<img src="images/RAG-QueryType.png" width="100%">

## File Structure

```
docs/
|   ├── analisi1.pdf        # Test pdf for querying

src/
│   ├── chunk_embed.py       # Tokenization, chunking, and embedding
│   ├── index.py             # Qdrant Vector DB wrapper
│   ├── retriever.py         # Retriever class to fetch relevant chunks
│   ├── rag_engine.py        # RAG class combining retriever + LLM
│   └── utils.py             # Docling markdown + summary replacements

images/
│   ├── screenshot.png       # Interface screenshot

output/
│   ├── output.md            # Raw markdown output from Docling

app.py                    # Main Streamlit app
README.md                 # You're reading it
```

---

## How It Works

1. **PDF Upload**: Users upload a PDF in the sidebar.
2. **Docling**: PDF is converted to markdown (with layout + tables + image data).
3. **Chunking + Embedding**:

   * Tokenized into 1024-token overlapping chunks.
   * Embedded using `nomic-embed-text-v1.5`.
4. **Indexing**: Embeddings are stored in a **Qdrant vector DB**.
5. **Querying**:

   * User queries are embedded.
   * Top-7 relevant chunks are retrieved using **dot-product similarity**.
   * These are passed to **GPT-5** for final answer generation.



## Demo (Docker Compose Setup)

1. **Install Docker + Docker Compose**  
   Follow the [official Docker installation guide](https://www.docker.com/get-started).

2. **Create your `.env` file in the project root**:

   ```env
   OPENAI_API_KEY=your_key_here
   OPENAI_MODEL=your_model_here
   ```

3. **Start the full multi-container app (frontend + backend + qdrant)**:

   ```bash
   docker compose up -d
   ```

   If you changed dependencies or the `Dockerfile`, rebuild:

   ```bash
   docker compose up --build -d
   ```

4. **Access services**:
   - App (Streamlit): `http://localhost:8501`
   - Backend health: `http://localhost:8000/health`
   - Qdrant dashboard: `http://localhost:6333/dashboard`

5. **Inspect logs**:

   ```bash
   docker compose logs --tail=200 backend
   docker compose logs --tail=200 frontend
   docker compose logs -f backend frontend
   ```

6. **Stop or restart**:

   ```bash
   docker compose down
   docker compose restart backend
   docker compose restart frontend
   ```

7. **Reset Qdrant data (destructive)**:

   ```bash
   docker compose down -v
   ```


## References

* [Docker](https://www.docker.com/get-started)
* [Docling](https://github.com/docling-project/docling)
* [Streamlit Chat UI](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)
* [Qdrant Vector DB](https://qdrant.tech/)
* [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
