# Vector DB and semantic search

import chromadb
from langchain_core.documents import Document
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
import streamlit as st
from sentence_transformers import CrossEncoder

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    ) 

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(
    all_splits: list[Document], 
    file_name: str,
    min_chunk_length: int = 30
):
    """
    Adds cleaned document chunks to ChromaDB vector collection.

    Args:
        all_splits (list): LangChain Document chunks.
        file_name (str): The source filename (used as unique ID prefix).
        min_chunk_length (int): Minimum chunk length to include (default: 30 chars).

    Raises:
        ChromaDBError: If vector upsert fails.
    """
    if not all_splits:
        st.warning(f"🚫 No valid content found in {file_name} — skipping.")
        return

    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    seen_chunks = set()

    for idx, split in enumerate(all_splits):
        text = split.page_content.strip()

        if len(text) < min_chunk_length:
            continue  # Skip very short chunks

        chunk_hash = hash(text)
        if chunk_hash in seen_chunks:
            continue  # Skip duplicate chunks
        seen_chunks.add(chunk_hash)

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
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        st.success(f"✅ Added {len(documents)} chunks from {file_name} to vector store.")
    except Exception as e:
        st.error(f"❌ Failed to upsert {file_name}: {str(e)}")

def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(
        query_texts=[prompt], 
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )
    
    # Add source file information to results
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
        prompt (str): User query.
        documents (list): List of candidate document strings.
        top_k (int): Number of top results to return (default: 3).

    Returns:
        Tuple[str, list[int]]: Concatenated top document text and their indices.

    Raises:
        RuntimeError: If cross-encoder model fails or no results are found.
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