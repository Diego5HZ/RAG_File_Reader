# Vector DB and semantic search

import chromadb
from langchain_core.documents import Document
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import streamlit as st
from sentence_transformers import CrossEncoder

# Persistent Chroma client initialization (reuse the client across function calls)
client = chromadb.PersistentClient(path="./demo-rag-chroma")

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    collection_name = "rag_app"  # Ensure the same collection name is used

    # Create or get the collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Debugging: Check if the collection is created or retrieved
    print(f"Collection created or retrieved: {collection.name}")
    return collection

def process_document_splits(all_splits: list[Document], file_name: str, min_chunk_length: int = 30):
    """
    Process document splits to prepare for vector upsert. Filters out short or duplicate chunks.
    
    Args:
        all_splits: List of LangChain Document chunks.
        file_name: The source filename (used for unique ID).
        min_chunk_length: Minimum length to include in vector store.
        
    Returns:
        documents, metadatas, ids: Lists of cleaned documents, metadata, and ids.
    """
    documents, metadatas, ids = [], [], []
    seen_chunks = set()

    for idx, split in enumerate(all_splits):
        text = split.page_content.strip()
        
        if len(text) < min_chunk_length:
            continue  # Skip short chunks

        chunk_hash = hash(text)
        if chunk_hash in seen_chunks:
            continue  # Skip duplicate chunks
        seen_chunks.add(chunk_hash)

        # Update metadata with additional info
        split.metadata.update({
            "source_file": file_name,
            "chunk_id": idx,
            "text_length": len(text),
        })

        documents.append(text)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    return documents, metadatas, ids

def add_to_vector_collection(all_splits: list[Document], file_name: str, min_chunk_length: int = 30):
    """
    Adds cleaned document chunks to ChromaDB vector collection.

    Args:
        all_splits: LangChain Document chunks.
        file_name: The source filename (used as unique ID).
        min_chunk_length: Minimum chunk length to include (default: 30 chars).
    """
    if not all_splits:
        st.warning(f"🚫 No valid content found in {file_name} — skipping.")
        return

    documents, metadatas, ids = [], [], []
    seen_chunks = set()

    for idx, split in enumerate(all_splits):
        text = split.page_content.strip()
        
        if len(text) < min_chunk_length:
            continue  # Skip short chunks

        chunk_hash = hash(text)
        if chunk_hash in seen_chunks:
            continue  # Skip duplicate chunks
        seen_chunks.add(chunk_hash)

        # Handle metadata: Convert list metadata into a valid format (e.g., string)
        headings_metadata = split.metadata.get("headings", [])
        
        if headings_metadata:
            headings_str = "\n".join([f"Level {heading['level']}: {heading['heading']}" for heading in headings_metadata])
            split.metadata["headings"] = headings_str  # Replace list with string
        else:
            split.metadata["headings"] = "No headings found"  # Handle empty list case

        # Check and convert all other metadata fields to valid types
        for key, value in split.metadata.items():
            if isinstance(value, list):
                split.metadata[key] = str(value) if value else "No data"

        # Update metadata with additional info (e.g., source file, chunk id, text length)
        split.metadata.update({
            "source_file": file_name,
            "chunk_id": idx,
            "text_length": len(text),
        })

        documents.append(text)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    if not documents:
        st.warning(f"⚠️ No valid text to add from {file_name}. All chunks were filtered.")
        return

    try:
        collection = get_vector_collection()
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        # After the successful upsert, update the processed files list
        st.session_state.processed_files = getattr(st.session_state, 'processed_files', [])
        if file_name not in st.session_state.processed_files:
            st.session_state.processed_files.append(file_name)

        st.success(f"✅ Added {len(documents)} chunks from {file_name} to vector store.")
        
    except Exception as e:
        st.error(f"❌ Failed to upsert {file_name}: {str(e)}")

def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents."""
    collection = get_vector_collection()
    results = collection.query(
        query_texts=[prompt], 
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    if results["metadatas"]:
        for i, metadata in enumerate(results["metadatas"][0]):
            results["documents"][0][i] = {
                "text": results["documents"][0][i],
                "source": metadata.get("source_file", "unknown"),
                "page": metadata.get("page", "unknown")
            }
    return results

def re_rank_cross_encoders(prompt: str, documents: list[str], top_k: int = 3) -> tuple[str, list[int]]:
    """
    Re-ranks documents using a cross-encoder model for improved semantic accuracy.

    Args:
        prompt: User query.
        documents: List of candidate document strings.
        top_k: Number of top results to return.

    Returns:
        Tuple[str, list[int]]: Concatenated top document text and their indices.
    """
    if not documents or not prompt.strip():
        st.warning("⚠️ Empty input passed to re-ranking. Skipping.")
        return "", []

    try:
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        encoder_model = CrossEncoder(model_name)

        # Score all docs against the prompt
        scores = encoder_model.predict([(prompt, doc) for doc in documents])
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        relevant_texts = [documents[i] for i in ranked_indices]
        relevant_text = "\n\n".join(relevant_texts)

        # Optional: Display top scores (for debugging)
        for i in ranked_indices:
            st.markdown(f"**Doc {i} Score:** {scores[i]:.4f}")

        return relevant_text, ranked_indices

    except Exception as e:
        st.error(f"Cross-encoder re-ranking failed: {str(e)}")
        return "", []
