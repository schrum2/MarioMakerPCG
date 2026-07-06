"""Evaluation metrics for a json of generated MM2 levels (the all_levels.json
from run_diffusion.py or evaluate_caption_adherence.py). The MM counterpart of
MarioDiffusion's evaluate_metrics.py. Reports:

  * Broken structure counts per feature (see util/mm2_metrics.py).
  * AMED_self: diversity of the output set.
  * AMED_real (with --real_json): distance from each level to the training set.
  * Caption adherence when entries carry prompts: score, perfect matches, and
    per-topic phrase targeting. Deterministic prompts only; --skip_adherence
    for LLM-captioned sets.
  * Same-source diversity when entries carry source metadata: max edit distance
    among scenes generated from different captions of one source scene.

Example:
    python evaluate_mm2_metrics.py --json MODEL-unconditional-samples-short\\all_levels.json --real_json datasets\\MM_LevelsAndCaptions-regular.json
"""
import argparse
import json
import os

import util.common_settings as common_settings
from MarioMaker_create_ascii_captions import build_id_to_char
from captions.MM2_caption_match import compare_captions, extract_topics
from util.mm2_metrics import (
    broken_structure_report,
    average_min_edit_distance,
    average_min_edit_distance_from_real,
    max_edit_distance,
    source_group_key,
)


def phrase_targeting_metrics(prompt_caption_pairs):
    """Precision/recall/F1 per topic across (prompt, generated caption) pairs,
    over the MM entity topics the pairs mention rather than a fixed keyword list."""
    extracted = [(extract_topics(p), extract_topics(c)) for p, c in prompt_caption_pairs]
    topics = set()
    for pt, ct in extracted:
        topics |= set(pt) | set(ct)

    stats = {}
    for topic in sorted(topics):
        tp = fp = tn = fn = 0
        for pt, ct in extracted:
            in_prompt = topic in pt
            in_caption = topic in ct
            if in_prompt and in_caption:
                tp += 1
            elif in_caption:
                fp += 1
            elif in_prompt:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        stats[topic] = {
            "true_positives": tp, "false_positives": fp,
            "true_negatives": tn, "false_negatives": fn,
            "precision": precision, "recall": recall, "f1_score": f1,
        }
    return stats


def broken_structure_metrics(scenes, id_to_char):
    report = broken_structure_report(scenes, id_to_char)
    overall = report.pop("_overall")
    num_scenes = overall["num_scenes"]
    for name, entry in report.items():
        entry["broken_percentage_of_feature"] = (
            entry["broken"] / entry["total"] * 100 if entry["total"] else 0.0)
        entry["scenes_with_broken_percentage"] = (
            entry["scenes_with_broken"] / num_scenes * 100 if num_scenes else 0.0)
    overall["scenes_with_any_broken_percentage"] = (
        overall["scenes_with_any_broken"] / num_scenes * 100 if num_scenes else 0.0)
    return {"per_feature": report, "overall": overall}


def source_group_metrics(data, use_gpu=True):
    """Max edit distance within each group of scenes that share a source scene."""
    groups = {}
    for entry in data:
        key = source_group_key(entry)
        if key is None or "scene" not in entry:
            continue
        groups.setdefault(key, []).append(entry["scene"])

    per_group = {}
    for key, scenes in groups.items():
        if len(scenes) < 2:
            continue
        med = max_edit_distance(scenes, use_gpu=use_gpu)
        if med is not None:
            per_group[str(key)] = {"num_scenes": len(scenes), "max_edit_distance": med}

    if not per_group:
        return None
    values = [g["max_edit_distance"] for g in per_group.values()]
    return {
        "num_groups": len(per_group),
        "average_max_edit_distance": sum(values) / len(values),
        "min_max_edit_distance": min(values),
        "max_max_edit_distance": max(values),
        "per_group": per_group,
    }


def evaluate(args):
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = [entry["scene"] for entry in data if "scene" in entry]
    if not scenes:
        raise ValueError(f"No scenes found in {args.json}")

    id_to_char = build_id_to_char(args.tileset)

    metrics = {"file_name": os.path.basename(args.json),
               "total_generated_levels": len(scenes)}

    print(f"Analyzing broken structures in {len(scenes)} scenes...")
    metrics["broken_structures"] = broken_structure_metrics(scenes, id_to_char)

    print("Calculating AMED_self...")
    avg, compared, skipped = average_min_edit_distance(scenes, use_gpu=not args.cpu)
    metrics["average_min_edit_distance"] = avg
    metrics["amed_self_levels_compared"] = compared
    metrics["amed_self_levels_skipped"] = skipped

    if args.real_json:
        print("Calculating AMED_real...")
        with open(args.real_json, "r", encoding="utf-8") as f:
            real_levels = [entry["scene"] for entry in json.load(f) if "scene" in entry]
        avg, perfect, compared, skipped = average_min_edit_distance_from_real(
            scenes, real_levels, use_gpu=not args.cpu)
        metrics["average_min_edit_distance_from_real"] = avg
        metrics["generated_vs_real_perfect_matches"] = perfect
        metrics["percent_perfect_matches"] = perfect / len(scenes) * 100
        metrics["amed_real_levels_compared"] = compared
        metrics["amed_real_levels_skipped"] = skipped

    pairs = [(entry["prompt"], entry["caption"]) for entry in data
             if entry.get("prompt") and entry.get("caption")]
    if pairs and not args.skip_adherence:
        print(f"Scoring caption adherence on {len(pairs)} prompt/caption pairs...")
        # Trust scores saved at generation time when present; otherwise rescore.
        scores = [entry["score"] if "score" in entry
                  else compare_captions(entry["prompt"], entry["caption"])
                  for entry in data if entry.get("prompt") and entry.get("caption")]
        perfect = sum(1 for s in scores if s >= 1.0)
        metrics["average_caption_score"] = sum(scores) / len(scores)
        metrics["perfect_caption_matches"] = perfect
        metrics["perfect_caption_match_percentage"] = perfect / len(scores) * 100
        metrics["phrase_targeting"] = phrase_targeting_metrics(pairs)
    elif pairs:
        print("Skipping caption adherence (--skip_adherence).")

    group_metrics = source_group_metrics(data, use_gpu=not args.cpu)
    if group_metrics:
        print(f"Computed same-source max edit distance over {group_metrics['num_groups']} groups.")
        metrics["same_source_diversity"] = group_metrics

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated Mario Maker 2 levels: broken structures, AMED, caption adherence, and same-source diversity.")
    parser.add_argument("--json", type=str, required=True, help="Generated levels json (all_levels.json)")
    parser.add_argument("--real_json", type=str, default=None, help="Training dataset json for AMED_real")
    parser.add_argument("--tileset", type=str, default=common_settings.MM2_TILESET, help="Tileset json used to encode the scenes")
    parser.add_argument("--output", type=str, default=None, help="Output json (default: evaluation_metrics.json next to --json)")
    parser.add_argument("--skip_adherence", action="store_true", help="Skip prompt-vs-caption scoring (use when the prompts are LLM captions rather than deterministic ones)")
    parser.add_argument("--override", action="store_true", help="Recompute even if the output file already exists")
    parser.add_argument("--cpu", action="store_true", help="Do not use the GPU for edit distance calculations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.json) or ".", "evaluation_metrics.json")
    if os.path.exists(args.output) and not args.override:
        print(f"Error: '{args.output}' already exists. Use --override to recompute.")
        exit(1)
    evaluate(args)
