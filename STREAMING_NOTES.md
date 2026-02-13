# Streaming Implementation Notes

This document explains, step by step, how streaming was implemented for the Streamlit UI and FastAPI backend.

## 1) Confirmed the original behavior

The Streamlit app originally called a blocking request (`/query`) that returned a full string. Even when `st.write_stream` was used, it received a complete string and rendered everything at once. This provided no perceived latency improvement.

## 2) Added a streaming endpoint in FastAPI

A new endpoint, `POST /query_stream`, was added in `main.py` using `StreamingResponse`. This endpoint emits Server-Sent Events (SSE) lines in the format:

```
data: <chunk>

```

At the end of the stream it emits:

```
data: [DONE]

```

If an error occurs, it emits:

```
data: [ERROR] <message>

```

### Why SSE

SSE allows a single HTTP response to stay open while the server pushes data to the client as it becomes available. It is simpler than WebSockets for one-way streaming.

## 3) Made the backend handle both string and generator outputs

The RAG layer (`RAG.query`) was already using `stream=True` from OpenAI, which returns a generator. That meant `query` no longer returned a single string.

Two helper functions were added in `main.py`:

- `_is_stream(result)`: detects if the returned object is an iterable generator rather than a string.
- `_consume_stream(result)`: consumes the generator and concatenates it into one string (for the non-stream `/query` endpoint).

This solved a crash where `_chunk_text` tried to call `len()` on a generator.

## 4) Streamed generator chunks directly in `/query_stream`

If `RAG.query` returns a generator, the SSE endpoint now iterates and yields each chunk directly:

```
for chunk in response_text:
    yield f"data: {chunk}\n\n"
```

If it returns a plain string, it is chunked and streamed in fixed sizes.

## 5) Prevented response buffering in FastAPI

To reduce buffering:

- `await asyncio.sleep(0)` was added after each yield to let the event loop flush.
- Headers were added to disable proxy buffering:
  - `Cache-Control: no-cache`
  - `X-Accel-Buffering: no`

These are standard SSE practices to force incremental delivery.

## 6) Updated the Streamlit client to parse SSE correctly

The Streamlit client (`app.py`) now uses `httpx` streaming and parses SSE lines manually:

- `iter_text(chunk_size=64)` receives partial data.
- A buffer is used to accumulate text until newlines arrive.
- Lines starting with `data: ` are extracted and yielded into `st.write_stream`.

This avoids waiting for large buffered chunks before showing text in the UI.

## 7) Added logging for visibility

Extensive logging was added to both the backend and frontend to confirm:

- When a stream starts and connects.
- When each chunk is emitted by the backend.
- When each chunk is received by the Streamlit client.
- When streams complete or errors occur.

These logs make it easy to see if data is flowing end-to-end.

## 8) Remaining limitation

The FastAPI endpoint now streams properly, but true low-latency depends on the underlying LLM call. If the LLM library does not yield tokens as they are generated, streaming will still appear “burst-like.” The current setup is ready for real-time streaming as long as `RAG.query` yields tokens incrementally.

## File Changes Summary

- `main.py`: Added `/query_stream`, stream detection, SSE headers, and chunk flushing.
- `app.py`: Added SSE parsing with buffered streaming and improved error handling.

