"""Shared plumbing for the experiment stages.

Every stage shells out to an existing repo-root script (train_mlm.py,
train_diffusion.py, run_diffusion.py, evaluate_*.py) rather than importing it, so
each script keeps its own heavy, lazily-loaded dependencies and its own CLI. This
module holds the subprocess runner, the text-encoder name resolver, the dataset
path derivations, and the argparse bits every stage shares.
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from . import paths

# --text_encoder aliases and their HuggingFace ids. "MLM" is special: it means
# "train our own encoder" (see mm2pipeline_experiment.mlm) rather than freeze one.
PRETRAINED_ENCODERS = {
    "minilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "gte": "Alibaba-NLP/gte-large-en-v1.5",
}


def resolve_text_encoder(name):
    """Return (kind, model_id) for a --text_encoder value. kind is "mlm" (train
    our own encoder, model_id None) or "pretrained" (freeze an existing one). A
    value that isn't "MLM" or a known alias is passed through as a raw HF id."""
    if name.lower() == "mlm":
        return "mlm", None
    return "pretrained", PRETRAINED_ENCODERS.get(name.lower(), name)


def resolve_num_tiles(num_tiles, tileset):
    """Tile-id count for a tileset: every glyph plus one appended padding/unknown
    id, matching how mm2pipeline_data.dataset encodes scenes. Resolving it from the
    tileset keeps training and evaluation on the same id space; --num_tiles wins
    when given."""
    if num_tiles is not None:
        return num_tiles
    path = paths.repo_path(tileset)
    if not path.is_file():
        path = Path(tileset)
    with open(path, encoding="utf-8") as f:
        tiles = json.load(f)["tiles"]
    return len(tiles) + 1


def dataset_paths(dataset, tokenizer=None, random_json=None):
    """Derive the split / tokenizer / random-caption paths from a captioned
    dataset, matching the names mm2pipeline_data.dataset split writes."""
    dataset = os.path.abspath(dataset)
    base, ext = os.path.splitext(dataset)
    return SimpleNamespace(
        dataset=dataset,
        train=f"{base}-train{ext}",
        validate=f"{base}-validate{ext}",
        test=f"{base}-test{ext}",
        tokenizer=os.path.abspath(tokenizer) if tokenizer else f"{base}_tokenizer.pkl",
        random=os.path.abspath(random_json) if random_json else f"{base}_random{ext}",
    )


def wipe(path):
    """Delete a directory (resolved against the repo root when relative, since
    that's the cwd every stage runs its scripts from). Used to clear regenerable
    model / sample folders before a fresh run."""
    p = Path(path)
    if not p.is_absolute():
        p = paths.REPO_ROOT / p
    if p.is_dir():
        shutil.rmtree(p)


def _run(head, args, *, auto_yes, check, label):
    cmd = [sys.executable, *head, *[str(a) for a in args]]
    print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
    # train_diffusion.py prompts before resuming from a checkpoint; feed it "y".
    stdin = "y\n" if auto_yes else None
    result = subprocess.run(cmd, cwd=str(paths.REPO_ROOT), input=stdin, text=True)
    if check and result.returncode != 0:
        sys.exit(f"ERROR: {label} failed (exit {result.returncode}).")
    return result.returncode


def run_script(script, args=(), *, auto_yes=False, check=True):
    """Run a repo-root python script under the current interpreter, from the repo
    root. On a non-zero exit, aborts the whole run (unless check=False)."""
    return _run([str(paths.repo_path(script))], args,
                auto_yes=auto_yes, check=check, label=script)


def run_module(module, args=(), *, check=True):
    """Run `python -m <module>` from the repo root (used for mm2pipeline_data)."""
    return _run(["-m", module], args, auto_yes=False, check=check, label=module)


def add_common_args(parser, *, seed_default=0):
    """The game / tileset / num_tiles / seed flags shared by every stage."""
    parser.add_argument("--game", default="MM", choices=["MM", "Mario"],
                        help="Which game the model targets (affects tile count and sample style).")
    parser.add_argument("--tileset", default="mm2_tileset_we.json",
                        help="Tileset JSON. Default: mm2_tileset_we.json.")
    parser.add_argument("--num_tiles", type=int, default=None,
                        help="Number of tile ids. Default: derived from the tileset "
                             "(glyphs + 1 padding id).")
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed.")
