from __future__ import annotations

import asyncio
import functools
import json
import os
import tempfile
import time
import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.retrieval.utils import convert_pdf_to_markdown
from src.retrieval.chunk_embed import EmbedData, save_embeddings, load_embeddings
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def _chunk_text(text: str, chunk_size: int = 48):
    if not text:
        return
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def _is_stream(result) -> bool:
    return isinstance(result, Iterable) and not isinstance(result, (str, bytes))


def _consume_stream(result) -> str:
    return "".join(result)


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
            logger.info("created new session session_id=%s", session_id)
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
    logger.info("ingest start session_id=%s filename=%s", session_id, file.filename)

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session = await _get_or_create_session(session_id)
    async with session.lock:
        if session.file_name == file.filename and session.rag is not None:
            logger.info("ingest cached session_id=%s filename=%s", session_id, file.filename)
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
            logger.info("loading cached embeddings path=%s", embeddings_path)
            embeddata = await _run_cpu(load_embeddings, embeddings_path)
        else:
            logger.info("building embeddings for filename=%s", file.filename)
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, file.filename)
                await _run_io(_write_bytes, file_path, file_bytes)
                markdown_text = await _run_cpu(convert_pdf_to_markdown, file_path)

            logger.info("chunking markdown")
            chunks = await _run_cpu(chunk_markdown, markdown_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="No content extracted from PDF.")

            embeddata = EmbedData(batch_size=8)
            logger.info("embedding chunks count=%d", len(chunks))
            await _run_cpu(embeddata.embed, chunks)
            logger.info("saving embeddings path=%s", embeddings_path)
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
            logger.info("creating collection=%s", database.collection_name)
            await _run_io(database.create_collection)
            logger.info("ingesting embeddings collection=%s", database.collection_name)
            await _run_io(database.ingest_data, embeddata)

        retriever = Retriever(database, embeddata=embeddata)
        rag = RAG(retriever)

        session.file_name = file.filename
        session.embeddata = embeddata
        session.database = database
        session.rag = rag
        session.last_access = time.time()

        logger.info("ingest complete session_id=%s filename=%s", session_id, file.filename)
        return {
            "status": "ready",
            "cached": embeddings_exist,
            "collection": database.collection_name,
        }


@app.post("/query")
async def query_document(request: QueryRequest) -> dict:
    logger.info("query start session_id=%s difficulty=%s", request.session_id, request.difficulty)
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

    if _is_stream(response_text):
        logger.info("query returned stream; consuming for non-stream response")
        response_text = _consume_stream(response_text)

    logger.info("query complete session_id=%s", request.session_id)
    return {"response": response_text}


@app.post("/query_stream")
async def query_document_stream(request: QueryRequest):
    logger.info("stream query start session_id=%s difficulty=%s", request.session_id, request.difficulty)
    session = await _get_session(request.session_id)
    if session is None or session.rag is None:
        raise HTTPException(status_code=400, detail="Session not initialized. Upload a PDF first.")

    async def event_stream():
        async with session.lock:
            try:
                response_text = await _run_io(
                    session.rag.query,
                    request.prompt,
                    request.difficulty,
                )
                session.last_access = time.time()
            except Exception as exc:
                logger.exception("stream query error session_id=%s", request.session_id)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return

        if _is_stream(response_text):
            logger.info("streaming generator response")
            for chunk in response_text:
                if not chunk:
                    continue
                logger.info("stream chunk size=%d", len(chunk))
                yield f"data: {json.dumps({'t': chunk})}\n\n"
                await asyncio.sleep(0)
        else:
            for chunk in _chunk_text(response_text):
                logger.info("stream chunk size=%d", len(chunk))
                yield f"data: {json.dumps({'t': chunk})}\n\n"
                await asyncio.sleep(0)
        yield f"data: {json.dumps({'done': True})}\n\n"
        logger.info("stream query done session_id=%s", request.session_id)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/reset")
async def reset_session(request: ResetRequest) -> dict:
    logger.info("reset start session_id=%s", request.session_id)
    session = await _get_session(request.session_id)
    if session is None or session.rag is None:
        return {"status": "missing"}

    async with session.lock:
        session.rag.last_question = None
        session.rag.conversation_history = []
        session.last_access = time.time()

    logger.info("reset complete session_id=%s", request.session_id)
    return {"status": "ok"}


def main():
    import uvicorn

    uvicorn.run("main:app", reload=True)

if __name__ == "__main__":
    main()
