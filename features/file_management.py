import json
import os
from features.utils import normalize_filename 

# Path to the JSON file where processed files will be stored
processed_files_path = "processed_files.json"

def load_processed_files():
    """Load the list of processed files from the JSON file."""
    if os.path.exists(processed_files_path):
        with open(processed_files_path, "r") as f:
            return json.load(f)
    return []

def save_processed_files(processed_files):
    """Save the list of processed files to the JSON file."""
    with open(processed_files_path, "w") as f:
        json.dump(processed_files, f)

def update_processed_files(file_name):
    """Add a file name to the processed files and update the JSON file."""
    processed_files = load_processed_files()
    if file_name not in processed_files:
        processed_files.append(file_name)
        save_processed_files(processed_files)

import datetime

def save_reasoning_metadata(file_name: str, prompt: str, reasoning: str, sources: list[str]):
    """
    Saves reasoning with metadata (prompt, sources, timestamp) to a JSON file.
    """
    safe_name = normalize_filename(file_name)
    output_dir = "reasoning_outputs"
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "file": file_name,
        "normalized_file": safe_name,
        "prompt": prompt,
        "sources_used": sources,
        "timestamp": datetime.datetime.now().isoformat(),
        "reasoning": reasoning
    }

    metadata_filename = os.path.join(output_dir, f"reasoning_{safe_name}.json")
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)