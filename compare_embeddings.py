#!/usr/bin/env python3
"""
compare_embeddings.py

Scans all Block2Vec (B2V*) and Skip-gram (Skip*) run directories produced by
train_block2vec.py / train_skipgram.py, reads each run's
embedding_analysis/per_tile_report.json + summary.txt, optionally loads the
raw trained embedding tensors for deeper diagnostics, and writes a single
human-readable comparison report (Markdown) plus a CSV of all metrics.

Usage (run from the directory that CONTAINS the B2V*/Skip* folders, e.g.
the MarioMakerPCG repo root):

    python compare_embeddings.py
    python compare_embeddings.py --root . --out_dir embedding_comparison
    python compare_embeddings.py --pattern "B2V*" "Skip*" --top_k 5

What this does NOT do: train anything, or modify any run directory. It only
reads existing per_tile_report.json files and (if present) raw embedding
tensors, and writes new files into --out_dir.

Why both JSON-based and raw-tensor-based metrics:
  - per_tile_report.json already contains per-tile diagnostics computed at
    train time (update counts, movement from init, top-1 neighbor sim). These
    are cheap and always available if the run finished.
  - Loading the *raw* embedding matrix lets us compute whole-space diagnostics
    that per_tile_report.json doesn't capture: effective rank / isotropy,
    the full pairwise similarity distribution (not just top-1), and how many
    tile pairs are near-duplicates -- all of which matter a lot for a
    downstream diffusion model that will condition on these vectors instead
    of one-hot tile ids.

If raw tensors can't be found/loaded for a run, that run is still scored
using JSON-only metrics, and the report says so explicitly so you know which
numbers are partial.
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback

import numpy as np


# --------------------------------------------------------------------------
# Optional torch import. We want this script to still run (in JSON-only mode)
# even on a machine without torch installed, though in practice you'll be
# running this right next to the training scripts where torch is available.
# --------------------------------------------------------------------------
try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    from safetensors.torch import load_file as load_safetensors
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False


# ==========================================================================
# Run discovery + identity parsing
# ==========================================================================

def parse_run_name(dirname):
    """Parse 'B2V08_03' -> {'method': 'block2vec', 'embedding_dim': 8,
    'subsample_threshold': 0.03}. Parse 'Skip16_10' similarly.
    Returns None if the name doesn't match the expected pattern -- caller
    should skip such directories with a warning rather than crash.
    """
    base = os.path.basename(dirname.rstrip(os.sep))
    if base.startswith("B2V"):
        method = "block2vec"
        rest = base[3:]
    elif base.startswith("Skip"):
        method = "skipgram"
        rest = base[4:]
    else:
        return None

    if "_" not in rest:
        return None
    dim_str, thresh_str = rest.split("_", 1)
    try:
        embedding_dim = int(dim_str)
        # YY means threshold 0.YY, e.g. "03" -> 0.03, "10" -> 0.10
        subsample_threshold = int(thresh_str) / 100.0
    except ValueError:
        return None

    return {
        "method": method,
        "embedding_dim": embedding_dim,
        "subsample_threshold": subsample_threshold,
        "run_name": base,
    }


def discover_runs(root, patterns):
    found = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(root, pattern))):
            if not os.path.isdir(path):
                continue
            identity = parse_run_name(path)
            if identity is None:
                print(f"[skip] '{path}' does not match expected naming (B2V##_## / Skip##_##)")
                continue
            found.append((path, identity))
    return found


# ==========================================================================
# Loading per-run JSON report + summary.txt
# ==========================================================================

def load_json_report(run_dir):
    json_path = os.path.join(run_dir, "embedding_analysis", "per_tile_report.json")
    if not os.path.exists(json_path):
        return None, f"missing {json_path}"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, f"failed to parse {json_path}: {e}"


def load_summary_text(run_dir):
    summary_path = os.path.join(run_dir, "embedding_analysis", "summary.txt")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r") as f:
        return f.read()


# ==========================================================================
# Loading raw embedding tensors
#
# train_skipgram.py saves, directly in output_dir:
#     embeddings.pt        (raw torch tensor, in_embed.weight, cpu)
#     model.safetensors    (full state dict)
#     config.json
#
# train_block2vec.py calls model.save_pretrained(output_dir) -- the exact
# file layout that produces wasn't in the files reviewed when this script
# was written, so we try several plausible filenames/keys and fall back
# gracefully. If you know your Block2Vec.save_pretrained format, the
# RAW_EMBEDDING_CANDIDATES list below is the place to add it.
# ==========================================================================

RAW_EMBEDDING_FILENAMES = [
    "embeddings.pt",            # skipgram convention; also try for block2vec
    "in_embed.pt",
    "embedding.pt",
]

SAFETENSORS_FILENAMES = [
    "model.safetensors",
]

# Plausible key names for the "in" embedding weight matrix inside a
# state_dict / safetensors file. We try these in order.
SAFETENSORS_KEY_CANDIDATES = [
    "in_embed.weight",
    "in_embed.weights",
    "embeddings.in_embed.weight",
    "module.in_embed.weight",
]


def _find_latest_checkpoint_dir(run_dir):
    """If no top-level embedding file is found, look for the
    highest-numbered checkpoint_epoch* dir as a fallback."""
    candidates = glob.glob(os.path.join(run_dir, "checkpoint_epoch*"))
    candidates = [c for c in candidates if os.path.isdir(c)]
    if not candidates:
        return None

    def epoch_num(path):
        name = os.path.basename(path)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else -1

    candidates.sort(key=epoch_num)
    return candidates[-1]


def load_raw_embeddings(run_dir, expected_vocab_size=None, expected_dim=None):
    """Try hard to find and load the (vocab_size, embedding_dim) input
    embedding matrix for a run. Returns (np.ndarray or None, note_str).

    note_str explains what was loaded or why nothing could be loaded, and is
    surfaced in the report so partial/JSON-only results are never silently
    presented as complete.
    """
    if not HAVE_TORCH:
        return None, "torch not installed in this environment -- skipped raw tensor load"

    search_dirs = [run_dir]
    latest_ckpt = _find_latest_checkpoint_dir(run_dir)
    if latest_ckpt is not None:
        search_dirs.append(latest_ckpt)

    # 1. Plain .pt tensor files (skipgram's embeddings.pt, etc.)
    for d in search_dirs:
        for fname in RAW_EMBEDDING_FILENAMES:
            fpath = os.path.join(d, fname)
            if os.path.exists(fpath):
                try:
                    tensor = torch.load(fpath, map_location="cpu", weights_only=False)
                    arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
                    if arr.ndim == 2:
                        return arr, f"loaded raw tensor from {os.path.relpath(fpath, run_dir)}"
                except Exception as e:
                    print(f"[warn] failed loading {fpath}: {e}")

    # 2. safetensors state dicts
    if HAVE_SAFETENSORS:
        for d in search_dirs:
            for fname in SAFETENSORS_FILENAMES:
                fpath = os.path.join(d, fname)
                if os.path.exists(fpath):
                    try:
                        state = load_safetensors(fpath)
                        for key in SAFETENSORS_KEY_CANDIDATES:
                            if key in state:
                                arr = state[key].detach().cpu().numpy()
                                return arr, f"loaded '{key}' from {os.path.relpath(fpath, run_dir)}"
                        # last resort: any 2D tensor matching expected shape
                        for key, val in state.items():
                            if val.ndim == 2 and "embed" in key.lower():
                                arr = val.detach().cpu().numpy()
                                return arr, f"loaded '{key}' (heuristic match) from {os.path.relpath(fpath, run_dir)}"
                    except Exception as e:
                        print(f"[warn] failed loading {fpath}: {e}")

    return None, "no loadable embedding tensor found (tried embeddings.pt, model.safetensors); falling back to JSON-only metrics"


# ==========================================================================
# Raw-tensor diagnostics
# ==========================================================================

def effective_rank(emb, eps=1e-12):
    """Participation-ratio-style effective rank of the embedding matrix.
    Computed from the singular value spectrum: erank = (sum(s))^2 / sum(s^2).
    This is 1.0 if all variance is in a single direction (totally collapsed,
    useless for a diffusion model to condition on) and approaches
    min(vocab_size, embedding_dim) if variance is spread evenly across all
    dimensions (well-used embedding space).
    """
    centered = emb - emb.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s2 = s ** 2
    denom = np.sum(s2) + eps
    return float((np.sum(s) ** 2) / denom) if denom > 0 else float("nan")


def pairwise_cosine_stats(emb, near_dup_threshold=0.95):
    """Off-diagonal pairwise cosine similarity stats for the whole embedding
    table. High mean similarity or many near-duplicate pairs both indicate
    the embedding space hasn't differentiated tiles well -- bad news for a
    diffusion model that needs distinct, informative tile representations.
    """
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    n = sims.shape[0]
    iu = np.triu_indices(n, k=1)
    off_diag = sims[iu]
    near_dup_pairs = int(np.sum(off_diag >= near_dup_threshold))
    total_pairs = len(off_diag)
    return {
        "mean_pairwise_cosine": float(np.mean(off_diag)),
        "median_pairwise_cosine": float(np.median(off_diag)),
        "max_pairwise_cosine": float(np.max(off_diag)) if total_pairs else float("nan"),
        "near_duplicate_pairs": near_dup_pairs,
        "near_duplicate_pair_fraction": float(near_dup_pairs / total_pairs) if total_pairs else 0.0,
        "near_dup_threshold": near_dup_threshold,
    }, sims


def norm_stats(emb):
    norms = np.linalg.norm(emb, axis=1)
    return {
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_cv": float(np.std(norms) / (np.mean(norms) + 1e-8)),  # coefficient of variation
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def find_near_duplicate_pairs(emb, tile_ids, threshold=0.95, max_report=25):
    """Returns a list of (tile_a, tile_b, cosine_sim) for pairs at/above
    threshold, sorted descending, capped at max_report for readability."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = norm @ norm.T
    n = sims.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                pairs.append((tile_ids[i], tile_ids[j], float(sims[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:max_report], len(pairs)


# ==========================================================================
# JSON-report-derived metrics (always available if the run finished)
# ==========================================================================

def parse_summary_undertrained_count(summary_text):
    """Pull the undertrained tile count out of summary.txt's free text, as a
    fallback cross-check against what we compute ourselves from the JSON."""
    if summary_text is None:
        return None
    if "(none flagged)" in summary_text:
        return 0
    count = 0
    in_section = False
    for line in summary_text.splitlines():
        if "UNDERTRAINED" in line:
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("- Tile"):
                count += 1
            elif line.strip() == "":
                break
    return count


def json_derived_metrics(report):
    per_tile = report.get("per_tile", [])
    vocab_size = report.get("vocab_size", len(per_tile))
    embedding_dim = report.get("embedding_dim")

    update_counts = np.array(
        [row.get("train_update_count", np.nan) for row in per_tile], dtype=float
    )
    dataset_counts = np.array(
        [row.get("dataset_center_count", np.nan) for row in per_tile], dtype=float
    )
    movement = np.array(
        [row.get("distance_from_init", np.nan) for row in per_tile], dtype=float
    )
    top1_sims = np.array(
        [row.get("top1_neighbor_cosine_sim", np.nan) for row in per_tile], dtype=float
    )
    norms = np.array(
        [row.get("embedding_norm", np.nan) for row in per_tile], dtype=float
    )

    has_updates = not np.all(np.isnan(update_counts))
    metrics = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
    }

    zero_update_tiles = []
    if has_updates:
        nonzero = update_counts[(update_counts > 0) & ~np.isnan(update_counts)]
        median_updates = float(np.median(nonzero)) if len(nonzero) else 0.0
        threshold = max(1.0, 0.05 * median_updates)
        zero_update_tiles = [
            int(row["tile_id"]) for row in per_tile
            if row.get("train_update_count", 0) == 0
        ]
        undertrained_tiles = [
            int(row["tile_id"]) for row in per_tile
            if row.get("train_update_count", 0) < threshold
        ]
        metrics.update({
            "total_updates": float(np.nansum(update_counts)),
            "median_updates_nonzero": median_updates,
            "min_updates": float(np.nanmin(update_counts)),
            "max_updates": float(np.nanmax(update_counts)),
            "max_min_update_ratio": float(np.nanmax(update_counts) / max(np.nanmin(update_counts), 1)),
            "undertrained_threshold": threshold,
            "undertrained_tile_count": len(undertrained_tiles),
            "undertrained_tile_ids": undertrained_tiles,
            "zero_update_tile_count": len(zero_update_tiles),
            "zero_update_tile_ids": zero_update_tiles,
        })

    has_movement = not np.all(np.isnan(movement))
    if has_movement:
        metrics.update({
            "movement_mean": float(np.nanmean(movement)),
            "movement_min": float(np.nanmin(movement)),
            "movement_max": float(np.nanmax(movement)),
            "low_movement_tile_count": int(np.sum(movement < 0.05 * np.nanmean(movement))),
        })

    has_top1 = not np.all(np.isnan(top1_sims))
    if has_top1:
        metrics.update({
            "top1_sim_mean": float(np.nanmean(top1_sims)),
            "top1_sim_max": float(np.nanmax(top1_sims)),
            "top1_sim_gt_095_count": int(np.sum(top1_sims > 0.95)),
        })

    has_norms = not np.all(np.isnan(norms))
    if has_norms:
        metrics.update({
            "json_norm_mean": float(np.nanmean(norms)),
            "json_norm_std": float(np.nanstd(norms)),
        })

    # Does dataset frequency rank correlate with embedding norm? Tiles that
    # are rare in the data AND have tiny norm are the ones most likely to be
    # "dead" / uninformative for the diffusion model.
    has_dataset_counts = not np.all(np.isnan(dataset_counts))
    if has_dataset_counts and has_norms:
        rare_tiles = [
            int(row["tile_id"]) for row in per_tile
            if row.get("dataset_center_count", 0) == 0
        ]
        metrics["rare_in_data_tile_count"] = len(rare_tiles)
        metrics["rare_in_data_tile_ids"] = rare_tiles

    return metrics


# ==========================================================================
# Scoring / ranking across runs
# ==========================================================================

# (metric_key, direction, weight)
# direction: "lower_better" or "higher_better"
# Weights are a starting point reflecting what matters for downstream
# diffusion-model conditioning: every tile needs a *distinct*, *trained*,
# *well-scaled* vector. None of this replaces actually looking at the
# per-tile detail -- it's meant to triage 16 runs down to a shortlist.
SCORING_METRICS = [
    ("undertrained_frac", "lower_better", 2.0),
    ("max_min_update_ratio_log", "lower_better", 1.0),
    ("mean_pairwise_cosine", "lower_better", 1.5),
    ("near_duplicate_pair_fraction", "lower_better", 1.5),
    ("effective_rank_frac", "higher_better", 1.5),
    ("norm_cv", "lower_better", 0.75),
    ("top1_sim_mean", "lower_better", 0.75),
]


def compute_score_inputs(run_record):
    """Derive the normalized scoring inputs (see SCORING_METRICS) from a
    run's raw json_metrics / raw_metrics dicts. Returns dict; missing values
    are left out (and that run is scored on a smaller metric subset, noted
    in the report)."""
    jm = run_record["json_metrics"]
    rm = run_record.get("raw_metrics") or {}
    out = {}

    vocab_size = jm.get("vocab_size") or 1
    if "undertrained_tile_count" in jm:
        out["undertrained_frac"] = jm["undertrained_tile_count"] / vocab_size
    if "max_min_update_ratio" in jm:
        out["max_min_update_ratio_log"] = np.log10(max(jm["max_min_update_ratio"], 1.0))
    if "top1_sim_mean" in jm:
        out["top1_sim_mean"] = jm["top1_sim_mean"]

    if "mean_pairwise_cosine" in rm:
        out["mean_pairwise_cosine"] = rm["mean_pairwise_cosine"]
    if "near_duplicate_pair_fraction" in rm:
        out["near_duplicate_pair_fraction"] = rm["near_duplicate_pair_fraction"]
    if "effective_rank" in rm and jm.get("embedding_dim"):
        out["effective_rank_frac"] = rm["effective_rank"] / jm["embedding_dim"]
    if "norm_cv" in rm:
        out["norm_cv"] = rm["norm_cv"]
    elif "json_norm_std" in jm and jm.get("json_norm_mean"):
        out["norm_cv"] = jm["json_norm_std"] / (jm["json_norm_mean"] + 1e-8)

    return out


def rank_runs(run_records):
    """Min-max normalize each available metric across runs, combine into a
    weighted composite score (higher = better), and attach per-run rank.
    Mutates run_records in place, adding 'score_inputs' and 'composite_score'.
    """
    all_inputs = {}
    for rec in run_records:
        rec["score_inputs"] = compute_score_inputs(rec)
        for k, v in rec["score_inputs"].items():
            all_inputs.setdefault(k, []).append(v)

    ranges = {}
    for key, vals in all_inputs.items():
        vals = np.array(vals, dtype=float)
        ranges[key] = (float(np.min(vals)), float(np.max(vals)))

    for rec in run_records:
        total_weight = 0.0
        score = 0.0
        contributions = {}
        for key, direction, weight in SCORING_METRICS:
            if key not in rec["score_inputs"]:
                continue
            lo, hi = ranges[key]
            val = rec["score_inputs"][key]
            if hi - lo < 1e-12:
                norm = 0.5  # all runs tied on this metric
            else:
                norm = (val - lo) / (hi - lo)
            if direction == "higher_better":
                contrib = norm
            else:
                contrib = 1.0 - norm
            score += weight * contrib
            total_weight += weight
            contributions[key] = contrib
        rec["composite_score"] = (score / total_weight) if total_weight > 0 else float("nan")
        rec["score_contributions"] = contributions
        rec["score_metric_count"] = len(contributions)

    run_records.sort(key=lambda r: (-(r["composite_score"] if not np.isnan(r["composite_score"]) else -1)))
    for rank, rec in enumerate(run_records, start=1):
        rec["rank"] = rank
    return run_records


# ==========================================================================
# Report generation
# ==========================================================================

def fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_csv(run_records, out_path):
    fieldnames = [
        "rank", "run_name", "method", "embedding_dim", "subsample_threshold",
        "composite_score", "score_metric_count",
        "vocab_size",
        "undertrained_tile_count", "zero_update_tile_count", "max_min_update_ratio",
        "movement_mean", "movement_min", "movement_max",
        "top1_sim_mean", "top1_sim_max", "top1_sim_gt_095_count",
        "raw_tensor_loaded", "raw_tensor_note",
        "effective_rank", "effective_rank_frac",
        "mean_pairwise_cosine", "near_duplicate_pairs", "near_duplicate_pair_fraction",
        "norm_mean", "norm_std", "norm_cv",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in run_records:
            jm = rec["json_metrics"]
            rm = rec.get("raw_metrics") or {}
            row = {
                "rank": rec["rank"],
                "run_name": rec["identity"]["run_name"],
                "method": rec["identity"]["method"],
                "embedding_dim": rec["identity"]["embedding_dim"],
                "subsample_threshold": rec["identity"]["subsample_threshold"],
                "composite_score": rec["composite_score"],
                "score_metric_count": rec["score_metric_count"],
                "vocab_size": jm.get("vocab_size"),
                "undertrained_tile_count": jm.get("undertrained_tile_count"),
                "zero_update_tile_count": jm.get("zero_update_tile_count"),
                "max_min_update_ratio": jm.get("max_min_update_ratio"),
                "movement_mean": jm.get("movement_mean"),
                "movement_min": jm.get("movement_min"),
                "movement_max": jm.get("movement_max"),
                "top1_sim_mean": jm.get("top1_sim_mean"),
                "top1_sim_max": jm.get("top1_sim_max"),
                "top1_sim_gt_095_count": jm.get("top1_sim_gt_095_count"),
                "raw_tensor_loaded": rec["raw_loaded"],
                "raw_tensor_note": rec["raw_note"],
                "effective_rank": rm.get("effective_rank"),
                "effective_rank_frac": (rm.get("effective_rank") / jm["embedding_dim"]) if rm.get("effective_rank") and jm.get("embedding_dim") else None,
                "mean_pairwise_cosine": rm.get("mean_pairwise_cosine"),
                "near_duplicate_pairs": rm.get("near_duplicate_pairs"),
                "near_duplicate_pair_fraction": rm.get("near_duplicate_pair_fraction"),
                "norm_mean": rm.get("norm_mean"),
                "norm_std": rm.get("norm_std"),
                "norm_cv": rm.get("norm_cv"),
            }
            writer.writerow(row)


def write_markdown_report(run_records, out_path, near_dup_threshold):
    n_runs = len(run_records)
    n_raw_loaded = sum(1 for r in run_records if r["raw_loaded"])

    lines = []
    lines.append("# Tile Embedding Comparison Report")
    lines.append("")
    lines.append(f"Compared **{n_runs}** runs ({n_raw_loaded} with raw embedding tensors loaded, "
                  f"{n_runs - n_raw_loaded} JSON-only). Purpose: select tile embeddings to replace "
                  f"one-hot tile encodings as conditioning input to a diffusion model.")
    lines.append("")
    lines.append("**How to read this report:** the composite score is a rough triage tool, not a "
                  "verdict -- it min-max-normalizes several metrics across the runs you ran and "
                  "combines them with fixed weights (see `SCORING_METRICS` in the script). It tells "
                  "you relative standing *within this batch*, not whether any of them are actually "
                  "good enough in an absolute sense. Read the per-run detail and the 'Should you "
                  "retrain?' section below before picking a winner.")
    lines.append("")

    # ---------------- ranking table ----------------
    lines.append("## Ranking (best to worst, by composite score)")
    lines.append("")
    lines.append("| Rank | Run | Method | Dim | Subsample thr. | Score | Undertrained tiles | Max/min update ratio | Mean top-1 sim | Mean pairwise cos | Eff. rank | Raw tensor? |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rec in run_records:
        jm = rec["json_metrics"]
        rm = rec.get("raw_metrics") or {}
        ident = rec["identity"]
        eff_rank_str = (
            f"{rm['effective_rank']:.2f}/{jm['embedding_dim']}"
            if rm.get("effective_rank") is not None and jm.get("embedding_dim")
            else "n/a"
        )
        lines.append(
            f"| {rec['rank']} | `{ident['run_name']}` | {ident['method']} | {ident['embedding_dim']} "
            f"| {ident['subsample_threshold']:.2f} | {fmt(rec['composite_score'], 3)} "
            f"| {jm.get('undertrained_tile_count', 'n/a')} / {jm.get('vocab_size', '?')} "
            f"| {fmt(jm.get('max_min_update_ratio'), 1)} "
            f"| {fmt(jm.get('top1_sim_mean'), 3)} "
            f"| {fmt(rm.get('mean_pairwise_cosine'), 3)} "
            f"| {eff_rank_str} "
            f"| {'yes' if rec['raw_loaded'] else 'no'} |"
        )
    lines.append("")

    # ---------------- grouped views ----------------
    lines.append("## Grouped views")
    lines.append("")
    for method in ["block2vec", "skipgram"]:
        subset = [r for r in run_records if r["identity"]["method"] == method]
        if not subset:
            continue
        subset_sorted = sorted(subset, key=lambda r: r["rank"])
        best = subset_sorted[0]
        lines.append(f"**Best {method} run:** `{best['identity']['run_name']}` "
                      f"(rank {best['rank']} overall, score {fmt(best['composite_score'], 3)})")
    lines.append("")

    for dim in sorted(set(r["identity"]["embedding_dim"] for r in run_records)):
        subset = [r for r in run_records if r["identity"]["embedding_dim"] == dim]
        subset_sorted = sorted(subset, key=lambda r: r["rank"])
        best = subset_sorted[0]
        lines.append(f"**Best dim={dim} run:** `{best['identity']['run_name']}` "
                      f"(rank {best['rank']} overall, score {fmt(best['composite_score'], 3)})")
    lines.append("")

    # ---------------- per-run detail ----------------
    lines.append("## Per-run detail")
    lines.append("")
    for rec in run_records:
        ident = rec["identity"]
        jm = rec["json_metrics"]
        rm = rec.get("raw_metrics") or {}
        lines.append(f"### `{ident['run_name']}`  (rank {rec['rank']} / {n_runs}, score {fmt(rec['composite_score'], 3)})")
        lines.append("")
        lines.append(f"- Method: {ident['method']}, embedding_dim={ident['embedding_dim']}, "
                      f"subsample_threshold={ident['subsample_threshold']:.2f}")
        lines.append(f"- Vocab size: {jm.get('vocab_size', 'n/a')}")
        if "undertrained_tile_count" in jm:
            lines.append(f"- Undertrained tiles: {jm['undertrained_tile_count']} "
                          f"({jm.get('zero_update_tile_count', 0)} with zero updates) "
                          f"-- ids: {jm.get('undertrained_tile_ids', [])}")
            lines.append(f"- Update count range: min={fmt(jm.get('min_updates'), 0)}, "
                          f"max={fmt(jm.get('max_updates'), 0)}, "
                          f"max/min ratio={fmt(jm.get('max_min_update_ratio'), 1)}x")
        if "rare_in_data_tile_ids" in jm and jm["rare_in_data_tile_ids"]:
            lines.append(f"- Tiles with ZERO occurrences in the dataset itself "
                          f"(not just undertrained -- structurally absent): {jm['rare_in_data_tile_ids']}")
        if "movement_mean" in jm:
            lines.append(f"- Movement from init: mean={fmt(jm.get('movement_mean'))}, "
                          f"min={fmt(jm.get('movement_min'))}, max={fmt(jm.get('movement_max'))}, "
                          f"barely-moved tiles={jm.get('low_movement_tile_count', 'n/a')}")
        if "top1_sim_mean" in jm:
            lines.append(f"- Top-1 neighbor cosine similarity: mean={fmt(jm.get('top1_sim_mean'))}, "
                          f"max={fmt(jm.get('top1_sim_max'))}, "
                          f"tiles with top1_sim > 0.95: {jm.get('top1_sim_gt_095_count', 'n/a')}")

        lines.append(f"- Raw embedding tensor: {'loaded -- ' + rec['raw_note'] if rec['raw_loaded'] else 'NOT loaded -- ' + rec['raw_note']}")
        if rec["raw_loaded"]:
            lines.append(f"  - Effective rank: {fmt(rm.get('effective_rank'), 2)} "
                         f"out of {jm.get('embedding_dim', '?')} dimensions "
                         f"({fmt((rm.get('effective_rank') or 0) / jm['embedding_dim'] * 100 if jm.get('embedding_dim') else None, 1)}% "
                         f"of available dimensionality actually used)")
            lines.append(f"  - Pairwise cosine similarity: mean={fmt(rm.get('mean_pairwise_cosine'))}, "
                         f"median={fmt(rm.get('median_pairwise_cosine'))}, max={fmt(rm.get('max_pairwise_cosine'))}")
            lines.append(f"  - Near-duplicate pairs (cos_sim >= {near_dup_threshold}): "
                         f"{rm.get('near_duplicate_pairs', 'n/a')} "
                         f"({fmt(rm.get('near_duplicate_pair_fraction', 0) * 100, 2)}% of all pairs)")
            lines.append(f"  - Embedding norm: mean={fmt(rm.get('norm_mean'))}, "
                         f"std={fmt(rm.get('norm_std'))}, CV={fmt(rm.get('norm_cv'))}")
            if rec.get("near_dup_examples"):
                examples = ", ".join(f"({a},{b}: {s:.3f})" for a, b, s in rec["near_dup_examples"][:10])
                lines.append(f"  - Worst near-duplicate tile pairs: {examples}")
        lines.append("")

    # ---------------- "should you retrain" guidance ----------------
    lines.append("## Should you retrain / what to look at next")
    lines.append("")
    flagged_runs = []
    for rec in run_records:
        jm = rec["json_metrics"]
        rm = rec.get("raw_metrics") or {}
        issues = []
        vocab_size = jm.get("vocab_size", 1) or 1
        # Note: a handful of zero-update tiles is expected and often benign --
        # this codebase's tilesets include genuinely rare/low-frequency tiles,
        # so only flag this when it affects a non-trivial slice of the vocab.
        if jm.get("zero_update_tile_count", 0) > 0.03 * vocab_size:
            issues.append(f"{jm['zero_update_tile_count']} tile(s) got zero training updates "
                          f"(>3% of vocab -- check rare_in_data_tile_ids; if these tiles are absent "
                          f"from the dataset itself, no retraining will fix this)")
        if jm.get("undertrained_tile_count", 0) > 0.1 * vocab_size:
            issues.append(f"{jm['undertrained_tile_count']}/{vocab_size} tiles flagged undertrained (>10% of vocab)")
        if rm.get("near_duplicate_pair_fraction", 0) > 0.02:
            issues.append(f"{rm.get('near_duplicate_pairs')} near-duplicate tile pairs "
                          f"(cos_sim >= {near_dup_threshold}) -- these tiles may be indistinguishable to a downstream model")
        if rm.get("effective_rank") is not None and jm.get("embedding_dim"):
            frac = rm["effective_rank"] / jm["embedding_dim"]
            if frac < 0.5:
                issues.append(f"effective rank is only {frac*100:.0f}% of the embedding_dim -- "
                              f"space is collapsed onto fewer dimensions than you allocated")
        if issues:
            flagged_runs.append((rec, issues))

    if not flagged_runs:
        lines.append("No structural red flags detected (zero-update tiles, large near-duplicate clusters, "
                      "or heavily collapsed dimensionality) in any run. The top-ranked run(s) above are "
                      "reasonable candidates, but composite score alone shouldn't be the final word -- "
                      "consider visually inspecting the PCA / similarity heatmap figures for the top "
                      "1-2 runs, and sanity-checking that visually/semantically similar tiles (e.g. "
                      "different pipe orientations, different question-block states) actually end up "
                      "near each other.")
    else:
        lines.append("The following runs have at least one structural concern worth addressing "
                      "**before** treating them as production-ready, even if their composite score "
                      "looks decent:")
        lines.append("")
        for rec, issues in flagged_runs:
            lines.append(f"- **`{rec['identity']['run_name']}`** (rank {rec['rank']}): " + "; ".join(issues))
        lines.append("")
        lines.append("General fixes worth considering if most/all runs share these problems "
                      "(rather than re-tuning one run in isolation):")
        lines.append("- **Zero-update / rare tiles**: these tiles are likely rare or absent in the "
                      "training data itself (check `rare_in_data_tile_ids` above) -- no amount of "
                      "subsample-threshold tuning fixes a tile that never appears. Consider whether "
                      "the diffusion model actually needs to represent these tiles at all, or whether "
                      "more source data containing them is available.")
        lines.append("- **Near-duplicate tile pairs**: if specific tile id pairs show up as "
                      "near-duplicates across *most* runs (check the per-run 'worst near-duplicate "
                      "pairs' lists), that's a signal those tiles are contextually almost "
                      "interchangeable in your training data, not a training bug -- decide whether "
                      "that's actually true for your tileset or whether the context window/patch "
                      "size needs to be larger to distinguish them.")
        lines.append("- **Collapsed effective rank**: try a smaller `--embedding_dim` (no benefit to "
                      "paying for unused dimensions) or check whether `--negative_samples` is too low "
                      "for the vocab size, which can make the model converge to a low-rank solution.")
    lines.append("")

    lines.append("## Notes on this report's limitations")
    lines.append("")
    lines.append("- The composite score weights (see `SCORING_METRICS` in `compare_embeddings.py`) "
                  "are a reasonable starting guess, not a validated formula. Adjust the weights "
                  "directly in the script if you have a stronger prior about which property matters "
                  "more for your diffusion model.")
    lines.append("- None of these metrics directly measure what ultimately matters: whether the "
                  "diffusion model trained *with* these embeddings produces better level output than "
                  "one-hot encoding (or a different embedding run). They're a cheap proxy you can "
                  "compute before paying for that much more expensive downstream experiment.")
    lines.append("- `train_update_count` measures how many gradient updates a tile's row received, "
                  "which is a function of dataset frequency *and* subsampling settings -- it is not "
                  "the same thing as embedding quality. A tile can be updated a lot and still end up "
                  "with a poor embedding, or updated rarely and still land somewhere reasonable if its "
                  "few occurrences were informative.")
    if n_runs - n_raw_loaded > 0:
        lines.append(f"- {n_runs - n_raw_loaded} run(s) did not have a loadable raw tensor, so their "
                      f"effective rank / pairwise similarity / near-duplicate metrics are missing and "
                      f"their composite score rests on fewer signals than the others. See the "
                      f"'Raw embedding tensor' line per run above for why.")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare tile embedding training runs.")
    parser.add_argument("--root", type=str, default=".",
                        help="Directory containing the B2V*/Skip* run folders (default: current dir).")
    parser.add_argument("--pattern", type=str, nargs="+", default=["B2V*", "Skip*"],
                        help="Glob pattern(s) for run directories, relative to --root.")
    parser.add_argument("--out_dir", type=str, default="embedding_comparison",
                        help="Where to write the report + CSV.")
    parser.add_argument("--top_k_neighbors", type=int, default=5,
                        help="(Unused placeholder for symmetry with training scripts; "
                             "neighbor counts are read from the existing per_tile_report.json.)")
    parser.add_argument("--near_dup_threshold", type=float, default=0.95,
                        help="Cosine similarity threshold above which two tiles are flagged as near-duplicates.")
    args = parser.parse_args()

    runs = discover_runs(args.root, args.pattern)
    if not runs:
        print(f"No run directories found under '{args.root}' matching {args.pattern}.")
        sys.exit(1)

    print(f"Found {len(runs)} run directories.")

    run_records = []
    for run_dir, identity in runs:
        print(f"--- {identity['run_name']} ---")
        report, err = load_json_report(run_dir)
        if report is None:
            print(f"  [skip] {err}")
            continue

        summary_text = load_summary_text(run_dir)
        json_metrics = json_derived_metrics(report)

        # cross-check our own undertrained count against summary.txt's, just
        # in case the JSON and the printed summary ever disagree (e.g. if a
        # report.json was hand-edited or regenerated separately)
        summary_undertrained = parse_summary_undertrained_count(summary_text)
        if (summary_undertrained is not None
                and "undertrained_tile_count" in json_metrics
                and summary_undertrained != json_metrics["undertrained_tile_count"]):
            print(f"  [warn] undertrained count mismatch: summary.txt says {summary_undertrained}, "
                  f"recomputed from JSON says {json_metrics['undertrained_tile_count']}")

        raw_emb, raw_note = load_raw_embeddings(
            run_dir,
            expected_vocab_size=json_metrics.get("vocab_size"),
            expected_dim=json_metrics.get("embedding_dim"),
        )

        raw_metrics = None
        near_dup_examples = None
        if raw_emb is not None:
            tile_ids = [row["tile_id"] for row in report.get("per_tile", [])]
            if len(tile_ids) != raw_emb.shape[0]:
                print(f"  [warn] raw tensor has {raw_emb.shape[0]} rows but JSON has "
                      f"{len(tile_ids)} tiles -- using raw tensor's own row indices as tile ids")
                tile_ids = list(range(raw_emb.shape[0]))
            try:
                raw_metrics = {}
                raw_metrics["effective_rank"] = effective_rank(raw_emb)
                pw_stats, _ = pairwise_cosine_stats(raw_emb, near_dup_threshold=args.near_dup_threshold)
                raw_metrics.update(pw_stats)
                raw_metrics.update(norm_stats(raw_emb))
                near_dup_examples, _ = find_near_duplicate_pairs(
                    raw_emb, tile_ids, threshold=args.near_dup_threshold
                )
            except Exception as e:
                print(f"  [warn] raw tensor diagnostics failed: {e}")
                traceback.print_exc()
                raw_metrics = None

        run_records.append({
            "run_dir": run_dir,
            "identity": identity,
            "json_metrics": json_metrics,
            "raw_loaded": raw_emb is not None,
            "raw_note": raw_note,
            "raw_metrics": raw_metrics,
            "near_dup_examples": near_dup_examples,
        })

    if not run_records:
        print("No runs had a readable per_tile_report.json. Nothing to compare.")
        sys.exit(1)

    rank_runs(run_records)

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "embedding_comparison_report.md")
    csv_path = os.path.join(args.out_dir, "embedding_comparison.csv")
    write_markdown_report(run_records, md_path, args.near_dup_threshold)
    write_csv(run_records, csv_path)

    print()
    print(f"Wrote report to: {md_path}")
    print(f"Wrote CSV to:    {csv_path}")
    print()
    print("Top 3 runs by composite score:")
    for rec in run_records[:3]:
        print(f"  {rec['rank']}. {rec['identity']['run_name']}  score={rec['composite_score']:.3f}")


if __name__ == "__main__":
    main()
