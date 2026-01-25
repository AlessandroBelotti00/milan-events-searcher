from __future__ import annotations

import asyncio
import functools
import os
import tempfile
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.retrieval.utils import convert_pdf_to_markdown
from src.retrieval.chunk_embed import chunk_markdown, EmbedData, save_embeddings, load_embeddings
from src.retrieval.index import QdrantVDB
from src.retrieval.retriever import Retriever
from src.retrieval.rag_engine import RAG

CPU_EXECUTOR = ThreadPoolExecutor(max_workers=max(4, os.cpu_count() or 1))
IO_EXECUTOR = ThreadPoolExecutor(max_workers=16)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        CPU_EXECUTOR.shutdown(wait=False)
        IO_EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Recipe Maker API", lifespan=lifespan)


@dataclass
class SessionState:
    session_id: str
    file_name: Optional[str] = None
    rag: Optional[RAG] = None
    embeddata: Optional[EmbedData] = None
    database: Optional[QdrantVDB] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_access: float = field(default_factory=time.time)


sessions: Dict[str, SessionState] = {}
sessions_lock = asyncio.Lock()


def _write_bytes(path: str, content: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(content)


async def _run_cpu(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(CPU_EXECUTOR, functools.partial(func, *args))


async def _run_io(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(IO_EXECUTOR, functools.partial(func, *args))


async def _get_or_create_session(session_id: str) -> SessionState:
    async with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            session = SessionState(session_id=session_id)
            sessions[session_id] = session
        session.last_access = time.time()
        return session


async def _get_session(session_id: str) -> Optional[SessionState]:
    async with sessions_lock:
        session = sessions.get(session_id)
        if session:
            session.last_access = time.time()
        return session


class QueryRequest(BaseModel):
    session_id: str
    prompt: str
    difficulty: str


class ResetRequest(BaseModel):
    session_id: str


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session = await _get_or_create_session(session_id)
    async with session.lock:
        if session.file_name == file.filename and session.rag is not None:
            return {
                "status": "ready",
                "cached": True,
                "collection": session.database.collection_name if session.database else None,
            }

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        name = os.path.splitext(file.filename)[0]
        embeddings_path = f"embeddings_{name}.pkl"
        embeddings_exist = os.path.isfile(embeddings_path)

        if embeddings_exist:
            embeddata = await _run_cpu(load_embeddings, embeddings_path)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file.filename)
                await _run_io(_write_bytes, file_path, file_bytes)
                markdown_text = await _run_cpu(convert_pdf_to_markdown, file_path)

            chunks = await _run_cpu(chunk_markdown, markdown_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No content extracted from PDF.")

            embeddata = EmbedData(batch_size=8)
            await _run_cpu(embeddata.embed, chunks)
            await _run_io(save_embeddings, embeddata, embeddings_path)

        if not embeddata.embeddings:
            raise HTTPException(status_code=400, detail="No embeddings generated.")

        database = QdrantVDB(
            collection_name=f"collection_{name}",
            vector_dim=len(embeddata.embeddings[0]),
            batch_size=7,
        )
        exists = await _run_io(database.client.collection_exists, database.collection_name)
        if not exists:
            await _run_io(database.create_collection)
            await _run_io(database.ingest_data, embeddata)

        retriever = Retriever(database, embeddata=embeddata)
        rag = RAG(retriever)

        session.file_name = file.filename
        session.embeddata = embeddata
        session.database = database
        session.rag = rag
        session.last_access = time.time()

        return {
            "status": "ready",
            "cached": embeddings_exist,
            "collection": database.collection_name,
        }


@app.post("/query")
async def query_document(request: QueryRequest) -> dict:
    session = await _get_session(request.session_id)
    if session is None or session.rag is None:
        raise HTTPException(status_code=400, detail="Session not initialized. Upload a PDF first.")

    async with session.lock:
        response_text = await _run_io(
            session.rag.query,
            request.prompt,
            request.difficulty,
        )
        session.last_access = time.time()

    return {"response": response_text}


@app.post("/reset")
async def reset_session(request: ResetRequest) -> dict:
    session = await _get_session(request.session_id)
    if session is None or session.rag is None:
        return {"status": "missing"}

    async with session.lock:
        session.rag.last_question = None
        session.rag.conversation_history = []
        session.last_access = time.time()

    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("main:app", reload=True)

if __name__ == "__main__":
    main()
