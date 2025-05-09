# Normalization, filename, future helpers

def normalize_filename(filename: str) -> str:
    """Normalize filenames for consistent storage"""
    return filename.translate(str.maketrans({'-': '_', '.': '_', ' ': '_'}))