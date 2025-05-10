# PDF headings, metadata, OCR

from streamlit.runtime.uploaded_file_manager import UploadedFile
import tempfile
from io import BytesIO
from typing import List
import streamlit as st
import os
import re
from pypdf import PdfReader
from PIL import Image
import pytesseract

# Improved metadata extraction
def extract_document_metadata(uploaded_file: UploadedFile) -> dict:
    """
    Extracts metadata like title, author, keywords, and subject from the PDF.
    """
    metadata = {"title": "", "author": "", "keywords": "", "subject": ""}
    
    try:
        pdf_stream = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_stream)
        
        # Extract metadata
        doc_metadata = reader.metadata
        metadata["title"] = doc_metadata.get("title", "").strip()
        metadata["author"] = doc_metadata.get("author", "").strip()
        metadata["keywords"] = doc_metadata.get("keywords", "").strip()
        metadata["subject"] = doc_metadata.get("subject", "").strip()
        
    except Exception as e:
        st.warning(f"⚠️ Failed to extract metadata: {e}")

    return metadata

# Improved heading detection with hierarchical structure
def extract_document_structure(uploaded_file: UploadedFile) -> dict:
    """
    Extracts the document structure with hierarchical headings.
    """
    structure = {"title": uploaded_file.name, "headings": []}
    
    try:
        pdf_stream = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_stream)
        
        title = reader.metadata.get("title", "")
        if title:
            structure["title"] = title.strip()

        raw_headings = []
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.split("\n")

            for line in lines:
                cleaned = line.strip()
                if len(cleaned) < 20 or len(cleaned) > 120:
                    continue  # Skip noise or long body text

                # Match headings with different formats (numbered, markdown, uppercase)
                if cleaned.isupper():
                    raw_headings.append(cleaned)
                elif re.match(r"^\d+[\.\)]?\s+[A-Z][a-zA-Z\s\-]+", cleaned):  # Numbered headings (e.g., 1. Introduction)
                    raw_headings.append(cleaned)
                elif re.match(r"^#+\s+[A-Z]", cleaned):  # Markdown-like
                    raw_headings.append(cleaned)

        # Convert to hierarchical structure (main section and subsections)
        headings = []
        for heading in raw_headings:
            if heading.count('.') == 1:  # e.g., 1. Main Section
                headings.append({"level": 1, "heading": heading})
            elif heading.count('.') == 2:  # e.g., 1.1 Subsection
                headings.append({"level": 2, "heading": heading})
            else:
                headings.append({"level": 3, "heading": heading})  # Further levels

        structure["headings"] = headings

    except Exception as e:
        st.warning(f"⚠️ Could not extract structure: {e}")

    return structure

# Enhanced OCR processing
def enhanced_ocr_processing(image_path: str) -> str:
    """
    Process images to enhance OCR accuracy by pre-processing them before applying Tesseract.
    """
    try:
        img = Image.open(image_path)
        
        # Pre-process the image (e.g., convert to grayscale, increase contrast, etc.)
        img = img.convert('L')  # Convert to grayscale
        img = img.point(lambda x: 0 if x < 200 else 255)  # Simple thresholding
        
        # Apply Tesseract OCR
        text = pytesseract.image_to_string(img)
        
        return text
    except Exception as e:
        st.warning(f"⚠️ OCR processing failed: {e}")
        return ""

# Extract scientific metadata (citations, formulas, and technical terms)
def extract_scientific_metadata(text: str) -> dict:
    """Extracts citations, formulas, and technical terms"""
    return {
        "citations": re.findall(r'\([A-Za-z]+,?\s\d{4}\)', text),
        "formulas": re.findall(r'\$[^$]+\$|\\begin{equation}.*?\\end{equation}', text, re.DOTALL),
        "technical_terms": re.findall(r'\b[A-Z][a-z]+\s(?:Theorem|Lemma|Algorithm|Equation)\b', text)
    }

# Extract geospatial entities (coordinates, locations)
def extract_geospatial_entities(text: str) -> dict:
    """
    Extract geographical entities like coordinates, country names, etc., from the text.
    """
    coordinates = re.findall(r'\b(\d{1,3}\.\d+)\s*(?:,|\s+)(\d{1,3}\.\d+)\b', text)
    return {"coordinates": coordinates}

# Figure extraction with OCR for embedded images
def extract_figures(uploaded_file: UploadedFile) -> List[dict]:
    """Memory-efficient figure extraction with Windows compatibility"""
    figures = []
    temp_pdf_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_pdf_path = tmp.name

        reader = PdfReader(temp_pdf_path)
        for i, page in enumerate(reader.pages):
            for image in page.images:
                with tempfile.NamedTemporaryFile(suffix=".png") as img_tmp:
                    img_tmp.write(image.data)
                    img_tmp.flush()
                    try:
                        img = Image.open(img_tmp.name)
                        figures.append({
                            "page": i + 1,
                            "ocr_text": pytesseract.image_to_string(img)
                        })
                    except Exception:
                        continue
    except Exception as e:
        st.warning(f"⚠️ Figure extraction failed for {uploaded_file.name}: {str(e)}")
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                st.warning(f"⚠️ Could not delete temp file {temp_pdf_path}: {e}")

    return figures
