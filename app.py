## Adapted from streamlit tutorial. Refrence link below:
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)


import streamlit as st
import os
import base64
import uuid
import time
import gc
import httpx
import logging
import json

# Configurazioni della pagina
st.set_page_config(
    page_title="Exam Trainer Agent",
    page_icon="./images/logo1.png"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _ingest_document(file_bytes: bytes, filename: str, session_id: str) -> dict:
    logger.info("ingest start filename=%s session_id=%s size=%d", filename, session_id, len(file_bytes))
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {"session_id": session_id}
    timeout = httpx.Timeout(600.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_api_url("/ingest"), data=data, files=files)
        response.raise_for_status()
        logger.info("ingest complete status=%s", response.status_code)
        return response.json()


def _query_backend(session_id: str, prompt: str, difficulty: str) -> str:
    logger.info("query start session_id=%s difficulty=%s", session_id, difficulty)
    payload = {"session_id": session_id, "prompt": prompt, "difficulty": difficulty}
    timeout = httpx.Timeout(120.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_api_url("/query"), json=payload)
        response.raise_for_status()
        logger.info("query complete status=%s", response.status_code)
        return response.json()["response"]


def _query_backend_stream(session_id: str, prompt: str, difficulty: str):
    logger.info("stream query start session_id=%s difficulty=%s", session_id, difficulty)
    payload = {"session_id": session_id, "prompt": prompt, "difficulty": difficulty}
    timeout = httpx.Timeout(120.0, connect=10.0, read=None)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", _api_url("/query_stream"), json=payload) as response:
            response.raise_for_status()
            logger.info("stream query connected status=%s", response.status_code)
            buffer = ""
            for chunk in response.iter_text(chunk_size=64):
                if not chunk:
                    continue
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip("\r")
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning("stream received non-json data=%r", data)
                            continue

                        if payload.get("done") is True:
                            logger.info("stream query done")
                            return

                        error = payload.get("error")
                        if error:
                            logger.error("stream query error=%s", error)
                            raise RuntimeError(error)

                        text = payload.get("t")
                        if text:
                            logger.info("stream chunk received size=%d", len(text))
                            yield text


def _reset_backend(session_id: str) -> None:
    logger.info("reset start session_id=%s", session_id)
    payload = {"session_id": session_id}
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        client.post(_api_url("/reset"), json=payload)
    logger.info("reset complete session_id=%s", session_id)


def reset_chat():
    logger.info("reset chat start")
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

    if st.session_state.get("id"):
        try:
            _reset_backend(str(st.session_state.id))
        except httpx.HTTPError:
            st.warning("Backend reset failed. The next query may reuse prior context.")

    st.success("Chat cleared. You can start a new question now.")
    logger.info("reset chat complete")

# Function to display the uploaded PDF in the app
def display_pdf(file_bytes):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="500" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


# Sidebar: Upload Document
with st.sidebar:
    st.image("./images/cluster_reply.png")

    st.markdown("<h1 style='text-align: center;'> Use Exam Trainer Agent to test yourself</h1>", unsafe_allow_html=True)
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("", type="pdf")

    # Difficulty slider: map 1,2,3 to Easy, Medium, Hard
    difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
    difficulty_level = st.slider(
        "Select question difficulty",
        min_value=1,
        max_value=3,
        value=2,  # default medium
        format="%d"
    )
    # Store selected difficulty in session_state
    st.session_state.difficulty = difficulty_map[difficulty_level]
    

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        file_key = f"{session_id}-{uploaded_file.name}"
        if file_key not in st.session_state.file_cache:
            status_placeholder = st.empty()
            status_placeholder.info("File uploaded successfully")
        
            time.sleep(2.5)  # Delay before switching message
            status_placeholder.info("Processing document...")
            progress_bar = st.progress(15)

            try:
                _ingest_document(file_bytes, uploaded_file.name, str(session_id))
            except httpx.HTTPError as exc:
                status_placeholder.error("Backend ingestion failed. Please retry.")
                st.exception(exc)
            else:
                status_placeholder = st.empty()
                st.success("Ready to Chat...")
                progress_bar.progress(100)
                st.session_state.file_cache[file_key] = True
                
        else:
            st.success("Ready to Chat...")  
            

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.button("Clear", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Show message history (preserved across reruns)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user query
if prompt := st.chat_input("Ask a question..."):
    logger.info("user prompt received length=%d", len(prompt))
    
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate RAG-based response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Thinking..."):
            try:
                logger.info("starting stream to backend")
                stream = _query_backend_stream(
                    str(session_id),
                    prompt,
                    st.session_state.difficulty,
                )
                response = st.write_stream(stream)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    response_text = "Please upload a PDF to initialize the RAG system first."
                    st.warning(response_text)
                else:
                    response_text = "Backend error while generating a response."
                    st.error(response_text)
                logger.exception("http status error during stream")
                response = st.write_stream([response_text])
            except httpx.HTTPError:
                response_text = "Backend connection failed. Please retry."
                st.error(response_text)
                logger.exception("http error during stream")
                response = st.write_stream([response_text])
            except Exception:
                response_text = "Unexpected error while streaming response."
                st.error(response_text)
                logger.exception("unexpected error during stream")
                response = st.write_stream([response_text])


            

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
