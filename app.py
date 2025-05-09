import streamlit as st
import chromadb
from features.call_model import *
from features.content_extraction import *
from features.document_loader import * 
from features.embedding_store import * 
from features.utils import *
from features.analysis import *
import networkx as nx

client = chromadb.PersistentClient(path="./chroma_db")

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

    uploaded_files = st.file_uploader(
        "**📑 Upload PDF files**", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload any document you want to query with AI."
    )

    process = st.button("⚡️ Process Files")

    st.markdown("---")
    if st.button("🗑️ Reset YMA Vector DB"):
        # 1. Clear ChromaDB collections
        collections = client.list_collections()
        for col in collections:
            client.delete_collection(name=col.name)
        
        # 2. Clear Streamlit session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("✅ All collections and session memory cleared from YMA.")


# -------- Processing Files --------
if uploaded_files and process:
    st.info("🔄 YMA is processing your documents...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    if 'concept_graph' not in st.session_state:
        st.session_state.concept_graph = nx.Graph()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        progress_bar.progress((i + 1) / len(uploaded_files))

        normalized_name = normalize_filename(uploaded_file.name)
        doc_structure = extract_document_structure(uploaded_file)
        figures = extract_figures(uploaded_file)
        all_splits = process_document(uploaded_file)

        if all_splits:
            add_to_vector_collection(all_splits, normalized_name)
            st.session_state.concept_graph = build_concept_graph(all_splits)

    st.success("✅ All documents successfully processed into YMA's memory.")

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
                response = call_llm(context=relevant_text, prompt=prompt)
                st.write_stream(response)
            else:
                st.warning("⚠️ No meaningful context retrieved. Please try uploading documents first.")
        else:
            st.warning("📭 No documents retrieved from the database. Upload and process documents first.")

