"""Path resolution for the mm2pipeline package.

All the data files the pipeline depends on (tilesets, the ascii->extended
converter, ...) live at the repository root, NOT inside this package. Code used
to find them relative to each script's own directory; now that the logic lives in
a package one level down, resolve everything from the repo root instead so moving
a module never breaks a lookup.
"""
from pathlib import Path

# mm2pipeline/paths.py -> mm2pipeline/ -> repo root
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def repo_path(*parts) -> Path:
    """Path to a file/folder at the repository root (e.g. repo_path('smb.json'))."""
    return REPO_ROOT.joinpath(*parts)


# Canonical training tileset — the source of truth for the ASCII vocabulary.
MM2_TILESET_PATH = repo_path("mm2_tileset_we.json")
EXTENDED_TILESET_PATH = repo_path("extended_tiles.json")
SMB_TILESET_PATH = repo_path("smb.json")
