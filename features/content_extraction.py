# PDF headings, metadata, OCR

from streamlit.runtime.uploaded_file_manager import UploadedFile
import tempfile
from io import BytesIO
from typing import List
import streamlit as st

from pypdf import PdfReader
def extract_document_structure(uploaded_file: UploadedFile) -> dict:
    """Extracts hierarchical structure (headings, sections) from PDF"""
    try:
        pdf_stream = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_stream)
        return {
            "title": reader.metadata.title or uploaded_file.name,
            "headings": [
                text for page in reader.pages 
                for text in page.extract_text().split('\n') 
                if 20 < len(text) < 100 and text.isupper()
            ]
        }
    except Exception:
        return {"title": uploaded_file.name, "headings": []}

import re
def extract_scientific_metadata(text: str) -> dict:
    """Extracts citations, formulas, and technical terms"""
    return {
        "citations": re.findall(r'\([A-Za-z]+,?\s\d{4}\)', text),
        "formulas": re.findall(r'\$[^$]+\$|\\begin{equation}[^\\]+\\end{equation}', text),
        "technical_terms": re.findall(r'\b[A-Z][a-z]+\s(?:Theorem|Lemma|Algorithm|Equation)\b', text)
    }

    
from PIL import Image
import pytesseract
def extract_figures(uploaded_file: UploadedFile) -> List[dict]:
    """Memory-efficient figure extraction"""
    figures = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            
            reader = PdfReader(tmp.name)
            for i, page in enumerate(reader.pages):
                for image in page.images:
                    with tempfile.NamedTemporaryFile(suffix=".png") as img_tmp:
                        img_tmp.write(image.data)
                        img_tmp.flush()
                        try:
                            img = Image.open(img_tmp.name)
                            figures.append({
                                "page": i+1,
                                "ocr_text": pytesseract.image_to_string(img)
                            })
                        except Exception as e:
                            continue
    except Exception as e:
        st.warning(f"Figure extraction failed for {uploaded_file.name}: {str(e)}")
    return figures