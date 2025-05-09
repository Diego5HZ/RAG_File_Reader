# Loads + splits PDFs into Document chunks

from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from typing import List
from langchain_core.documents import Document
import tempfile
import os
import re

def process_document(uploaded_file: UploadedFile) -> List[Document]:
    """Processes an uploaded PDF into cleaned, structured chunks."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        loader = PyMuPDFLoader(temp_path)
        raw_docs = loader.load()

        text_splitter = ScientificTextSplitter()
        cleaned_docs = text_splitter.split_documents(raw_docs)

        return cleaned_docs

    except Exception as e:
        st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
        return []
    finally:
        try:
            os.unlink(temp_path)
        except (PermissionError, FileNotFoundError, UnboundLocalError):
            pass


class ScientificTextSplitter(RecursiveCharacterTextSplitter):
    """Enhanced splitter for scientific/technical documents."""

    def __init__(self):
        super().__init__(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n## ",  # Markdown headings
                "\n\n",     # Paragraphs
                "\n• ",     # Bullets
                "\.\s",     # Sentences
                "\n",       # Lines
                " ",        # Words
                ""          # Fallback
            ],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Cleans and splits raw LangChain Documents"""
        final_docs = []
        for doc in documents:
            cleaned = self._preprocess_text(doc.page_content)
            splits = self.split_text(cleaned)
            for chunk in splits:
                if chunk.strip():
                    final_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        return final_docs

    def _preprocess_text(self, text: str) -> str:
        """Custom text cleaning for scientific docs"""
        text = re.sub(r'Fig\.\s?\d+|Figure\s?\d+', '', text)  # Remove figure refs
        text = re.sub(r'\n+', '\n', text)  # Collapse multiple line breaks
        text = re.sub(r'\s{2,}', ' ', text)  # Remove extra spaces
        text = re.sub(r'^(\d{1,3}\s)', '', text, flags=re.MULTILINE)  # Remove line numbers
        text = re.sub(r'(References|Bibliography)[\s\S]*$', '', text, flags=re.IGNORECASE)  # Remove reference sections
        return text.strip()