"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Make the helpers package importable without installing the project
HELPERS_DIR = Path(__file__).parent / "notebooks" / "02-generate-embeddings"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))
