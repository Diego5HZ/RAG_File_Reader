# Normalization, filename, future helpers

def normalize_filename(filename: str) -> str:
    """Normalize filenames for consistent storage"""
    return filename.translate(str.maketrans({'-': '_', '.': '_', ' ': '_'}))

import hashlib

def hash_file_content(file) -> str:
    """Returns a short hash of the file content for deduplication."""
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()

import re

def clean_text(text: str) -> str:
    # Join broken lines (remove single newlines)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)

    # Optional: remove isolated characters (may delete acronyms though)
    text = re.sub(r'\b([a-zA-Z])\b', '', text)

    # Join words split by artificial spacing
    text = re.sub(r'(\b\w)\s+(\w\b)', r'\1\2', text)

    return text.strip()
