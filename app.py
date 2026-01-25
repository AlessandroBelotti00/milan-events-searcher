## Adapted from streamlit tutorial. Refrence link below:
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)


import streamlit as st
import os
import base64
import uuid
import time
import gc
import httpx

# Configurazioni della pagina
st.set_page_config(
    page_title="Exam Trainer Agent",
    page_icon="./images/logo1.png"
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def _ingest_document(file_bytes: bytes, filename: str, session_id: str) -> dict:
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {"session_id": session_id}
    timeout = httpx.Timeout(600.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_api_url("/ingest"), data=data, files=files)
        response.raise_for_status()
        return response.json()


def _query_backend(session_id: str, prompt: str, difficulty: str) -> str:
    payload = {"session_id": session_id, "prompt": prompt, "difficulty": difficulty}
    timeout = httpx.Timeout(120.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_api_url("/query"), json=payload)
        response.raise_for_status()
        return response.json()["response"]


def _reset_backend(session_id: str) -> None:
    payload = {"session_id": session_id}
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        client.post(_api_url("/reset"), json=payload)


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

    if st.session_state.get("id"):
        try:
            _reset_backend(str(st.session_state.id))
        except httpx.HTTPError:
            st.warning("Backend reset failed. The next query may reuse prior context.")

    st.success("Chat cleared. You can start a new question now.")

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
    
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate RAG-based response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
    
        with st.spinner("Thinking..."):
            response_text = ""
            try:
                response_text = _query_backend(
                    str(session_id),
                    prompt,
                    st.session_state.difficulty,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 400:
                    response_text = "Please upload a PDF to initialize the RAG system first."
                    st.warning(response_text)
                else:
                    response_text = "Backend error while generating a response."
                    st.error(response_text)
            except httpx.HTTPError as exc:
                response_text = "Backend connection failed. Please retry."
                st.error(response_text)
                st.exception(exc)
            if response_text:
                message_placeholder.markdown(response_text)

            

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
