## Adapted from streamlit tutorial. Refrence link below:
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps)


import streamlit as st
import os
import base64
import tempfile
import uuid
import time
import gc
import nest_asyncio 
nest_asyncio.apply()

from src.retrieval.utils import convert_pdf_to_markdown
from src.retrieval.chunk_embed import chunk_markdown, EmbedData, save_embeddings, load_embeddings
from src.retrieval.index import QdrantVDB
from src.retrieval.retriever import Retriever
from src.retrieval.rag_engine import RAG
from llama_index.core import Settings

# Configurazioni della pagina
st.set_page_config(
    page_title="Exam Trainer Agent",
    page_icon="./images/logo1.png"
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

    # Reset the current question in RAG
    rag = st.session_state.get("rag")
    if rag:
        rag.last_question = None
        rag.conversation_history = []

    st.success("Chat cleared. You can start a new question now.")

# Function to display the uploaded PDF in the app
def display_pdf(file):
    st.markdown("### 📄 PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
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
        file_key = f"{session_id}-{uploaded_file.name}"
        if file_key not in st.session_state.file_cache:
            status_placeholder = st.empty()
            status_placeholder.info("📥 File uploaded successfully")
        
            time.sleep(2.5)  # Delay before switching message
            name = uploaded_file.name.rsplit('.', 1)[0]

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print(f"Temporary file path: {file_path}")
                # Save uploaded file to temp dir
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                status_placeholder.info("Identifying document layout...")
                progress_bar = st.progress(10)

                found = any(
                    os.path.isfile(f) and f.startswith(f"embeddings_{name}" + '.')
                    for f in os.listdir('.')
                )

                if not found:
                    # Convert to markdown
                    markdown_text = convert_pdf_to_markdown(file_path)
                    st.session_state.markdown_text = markdown_text

                    status_placeholder.info("Generating embeddings...")
                    progress_bar.progress(50)
                    
                    chunks = chunk_markdown(markdown_text)
                    st.session_state.chunks = chunks

                    embeddata = EmbedData(batch_size=8)
                    embeddata.embed(chunks)
                    save_embeddings(embeddata, f"embeddings_{name}.pkl")

                    st.session_state.embeddata = embeddata

                    status_placeholder.info("Indexing the document...")
                    progress_bar.progress(80)
                
                else:
                    # se avevo già calcolato l'embeddings lo ricarico invece di ricalcolarmelo
                    embeddata = load_embeddings(f"embeddings_{name}.pkl")
                
                database = QdrantVDB(collection_name=f"collection_{name}", vector_dim=len(embeddata.embeddings[0]), batch_size=7)
                if database.client.collection_exists(f"collection_{name}"):
                    status_placeholder.info("Collection exists — loading existing index.")
                else:   
                    status_placeholder.info("Collection does NOT exist — creating new index.")
                    database.create_collection()
                    database.ingest_data(embeddata)


                st.session_state.database= database

                # After vector DB and embeddata have been defined...
                retriever = Retriever(database, embeddata=embeddata)
                rag = RAG(retriever)
                st.session_state.rag = rag
                status_placeholder = st.empty()
                st.success("Ready to Chat...")
                progress_bar.progress(100)
                st.session_state.file_cache[file_key] = True
                
        else:
            st.success("Ready to Chat...")  
            

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.button("Clear ↺", on_click=reset_chat)

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
        
            rag = st.session_state.get("rag")   

            if rag is None:
                st.warning("Please upload a PDF to initialize the RAG system first.")
            else:
                response_text = rag.query(prompt, difficulty=st.session_state.difficulty)
                message_placeholder.markdown(response_text)

        response = st.write_stream(response_text)
    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response_text})


