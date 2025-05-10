import streamlit as st
import chromadb
from features.call_model import *
from features.content_extraction import *
from features.document_loader import *
from features.embedding_store import * 
from features.utils import *
from features.analysis import *
from features.file_management import *  # For file tracking

import time
import json
import os

REASONING_LOG_FILE = "reasoning_log.json"

def update_reasoning_log(file_name: str, reasoning_file: str):
    """Log the reasoning file name associated with a document."""
    if os.path.exists(REASONING_LOG_FILE):
        with open(REASONING_LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[file_name] = reasoning_file

    with open(REASONING_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_reasoning_log() -> dict:
    """Load the reasoning log if it exists."""
    if os.path.exists(REASONING_LOG_FILE):
        with open(REASONING_LOG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_reasoning_to_file(file_name: str, reasoning: str):
    safe_name = normalize_filename(file_name)
    output_dir = "reasoning_outputs"
    os.makedirs(output_dir, exist_ok=True)
    reasoning_filename = os.path.join(output_dir, f"reasoning_{safe_name}.txt")

    with open(reasoning_filename, "w", encoding="utf-8") as file:
        file.write(reasoning)

    # Track it
    update_reasoning_log(file_name, reasoning_filename)
    st.success(f"✅ Reasoning saved to `{reasoning_filename}`")

client = chromadb.Client()

# Initialize processed_files in session state, using the file for persistence
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = load_processed_files()

# Initialize processed_hashes to keep track of already processed files
if 'processed_hashes' not in st.session_state:
    st.session_state.processed_hashes = set()

# -------- Page Config --------
st.set_page_config(
    page_title="YMA | Your Management Assistant",
    page_icon="🐿️",
    layout="wide"
)

# -------- Custom YMA Header --------
st.markdown(""" 
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="color: #1F4E79;">🐿️ YMA - Your Management Assistant</h1>
        <p style="font-size: 1.1em; color: #555;">
            Upload your PDFs and ask questions – powered by local AI and semantic search.
        </p>
    </div>
    <hr style="margin-top: -10px;">
""", unsafe_allow_html=True)

# -------- Sidebar: Upload & Reset --------
with st.sidebar:
    st.markdown("## ⚙️ YMA Settings")
    st.markdown("Use this panel to upload your documents and manage the database.")

    # File uploader with a dynamic key to reset the uploader after processing
    uploaded_files = st.file_uploader(
        "**📑 Upload PDF files**", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload any document you want to query with AI.",
        key="file_uploader"  # key used to reset the uploader
    )

    process = st.button("⚡️ Process Files")

    # ─── Reset Button ────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Reset YMA Vector DB"):
        # Delete all collections in ChromaDB
        collections = client.list_collections()
        for col in collections:
            client.delete_collection(name=col.name)
        
        # Clear processed files and hashes from session state
        st.session_state.processed_files = []  # Clear history of processed files
        st.session_state.processed_hashes = set()  # Clear processed hashes

        # Clear the processed files JSON to reset file history
        processed_files_path = "processed_files.json"
        if os.path.exists(processed_files_path):
            os.remove(processed_files_path)  # Delete the processed files JSON file

        # Optionally, clear any other session state you need
        for key in list(st.session_state.keys()):
            if key not in ['processed_files', 'processed_hashes']:  # Keep these two intact if needed
                del st.session_state[key]

        st.success("✅ All collections in the vector DB have been cleared. Processed files and history have been reset.")

# -------- Processing Files --------
if uploaded_files and process:
    st.info("🔄 YMA is processing your documents...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Store uploaded files temporarily in a list
    processed_file_names = []

    for i, uploaded_file in enumerate(uploaded_files):
        file_hash = hash_file_content(uploaded_file)

        if file_hash in st.session_state.processed_hashes:
            st.warning(f"⚠️ File `{uploaded_file.name}` already processed. Skipping.")
            continue
        st.session_state.processed_hashes.add(file_hash)

        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        progress_bar.progress((i + 1) / len(uploaded_files))

        normalized_name = normalize_filename(uploaded_file.name)
        figures = extract_figures(uploaded_file)  # Extract figures from the file
        all_splits = process_document(uploaded_file)  # Process document and split into chunks

        if all_splits:
            # Add the processed chunks to the vector collection and concept graph
            add_to_vector_collection(all_splits, normalized_name)
            time.sleep(2)

            # Save the file name to the processed list and persist it
            update_processed_files(uploaded_file.name)

            # Dynamically update the history of uploaded files in the session state
            st.session_state.processed_files = load_processed_files()  # Refresh the processed files from the file

            # Store processed file names to refresh UI
            processed_file_names.append(uploaded_file.name)

    st.success("✅ All documents successfully processed into YMA's memory.")

# -------- History of Uploaded Files (Body Section) --------
st.markdown("## 📂 History of Uploaded Files")
st.write("Here are the uploaded files that have been processed:")

if st.session_state.processed_files:
    for file in st.session_state.processed_files:
        st.markdown(f"- {file}")
else:
    st.write("No files processed yet.")

# -------- Q&A Section --------
st.markdown("## 🧠 Ask YMA a Question")
prompt = st.text_area("📥 What do you want to know from the uploaded documents?")

ask = st.button("🔎 Ask YMA")

if ask and prompt:
    with st.spinner("🤖 YMA is thinking..."):
        results = query_collection(prompt)
        docs = results.get("documents", [[]])[0]

        if docs:
            context = [doc["text"] for doc in docs]
            relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context)

            if relevant_text.strip():  # Additional safeguard
                response_stream = call_llm(context=relevant_text, prompt=prompt)
                final_response = ""
                response_chunks = []
                for chunk in response_stream:
                    response_chunks.append(chunk.replace('\n', ' ').strip())

                final_response = ' '.join(response_chunks).strip()

                # Render once, cleanly
                st.markdown("### 🧠 YMA's Response")
                st.markdown(final_response, unsafe_allow_html=True)

                # Save clean text
                top_source_file = docs[0].get("source", "unknown_file")
                save_reasoning_to_file(top_source_file, final_response)
                save_reasoning_metadata(
                    file_name=top_source_file,
                    prompt=prompt,
                    reasoning=final_response,
                    sources=[doc.get("source", "unknown") for doc in docs]
                )

            else:
                st.warning("⚠️ No meaningful context retrieved. Please try uploading documents first.")
        else:
            st.warning("📭 No documents retrieved from the database. Upload and process documents first.")
