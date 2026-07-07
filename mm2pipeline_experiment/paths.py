"""Path resolution for the mm2pipeline_experiment package.

The training / evaluation scripts and the tilesets live at the repo root, not in
the package, so every stage resolves them from there via repo_path().
"""
from pathlib import Path

# mm2pipeline_experiment/paths.py -> mm2pipeline_experiment/ -> repo root
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def repo_path(*parts) -> Path:
    """Path to a file/folder at the repository root (e.g. repo_path('train_mlm.py'))."""
    return REPO_ROOT.joinpath(*parts)


# The tileset the whole project trains against (see mm2pipeline_data.tiles).
MM2_TILESET_PATH = repo_path("mm2_tileset_we.json")
