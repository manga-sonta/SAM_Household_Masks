#!/usr/bin/env python3
"""
Analyze consistency results: group by augmentation type, mask count drift,
small/medium/large object stability, oversegmentation on original set,
and per-scene-class stats. Outputs a single JSON for plotting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scene_class_from_stem(stem: str) -> str:
    """e.g. bedroom__375 -> bedroom, kitchen__cdmc1164 -> kitchen."""
    if "__" in stem:
        return stem.split("__")[0]
    return "other"


def run(
    results_path: Path,
    original_metadata_dir: Path,
    analysis_output_path: Path,
    size_percentiles: tuple[float, float] = (33.0, 66.0),
) -> dict:
    results_path = Path(results_path)
    original_metadata_dir = Path(original_metadata_dir)
    with open(results_path) as f:
        results = json.load(f)

    # Load all original metadata (stem -> metadata)
    meta_by_stem = {}
    for p in original_metadata_dir.glob("*.json"):
        with open(p) as f:
            meta_by_stem[p.stem] = json.load(f)

    per_pair = results["per_pair"]

    # Per-mask (area, best_iou, augmentation, scene) for scatter and size bins
    mask_level = []
    # By augmentation type
    by_aug: dict[str, dict] = {}
    # Mask count drift per pair (and by aug)
    count_drifts: list[dict] = []
    # Oversegmentation: from original metadata only (all images)
    mask_count_per_image = []
    mask_areas_flat = []

    for stem, meta in meta_by_stem.items():
        mask_count_per_image.append(meta["num_masks"])
        for m in meta["masks"]:
            mask_areas_flat.append(m["area"])

    # Percentiles for size bins (over all original mask areas)
    areas_arr = np.array(mask_areas_flat)
    p33 = float(np.percentile(areas_arr, size_percentiles[0]))
    p66 = float(np.percentile(areas_arr, size_percentiles[1]))

    for pair in per_pair:
        orig_stem = pair["original_stem"]
        aug_type = pair["augmentation"]
        best_ious = pair.get("best_ious", [])
        n_orig = pair["num_original"]
        n_aug = pair["num_augmented"]
        drift = n_aug - n_orig
        scene = scene_class_from_stem(orig_stem)

        count_drifts.append({
            "augmentation": aug_type,
            "original_stem": orig_stem,
            "scene": scene,
            "drift": drift,
            "num_original": n_orig,
            "num_augmented": n_aug,
        })

        if aug_type not in by_aug:
            by_aug[aug_type] = {
                "mean_ious": [],
                "survival_050": [],
                "survival_070": [],
                "count_drifts": [],
                "n_pairs": 0,
            }
        by_aug[aug_type]["mean_ious"].append(pair["mean_best_iou"])
        by_aug[aug_type]["survival_050"].append(pair["survival_rate_050"])
        by_aug[aug_type]["survival_070"].append(pair["survival_rate_070"])
        by_aug[aug_type]["count_drifts"].append(drift)
        by_aug[aug_type]["n_pairs"] += 1

        # Mask-level: need areas for this image (same order as best_ious)
        meta = meta_by_stem.get(orig_stem)
        if meta and best_ious:
            areas = [m["area"] for m in meta["masks"]]
            n = min(len(areas), len(best_ious))
            for i in range(n):
                area = areas[i]
                iou = best_ious[i]
                size_bin = "small" if area < p33 else ("medium" if area < p66 else "large")
                mask_level.append({
                    "area": area,
                    "best_iou": iou,
                    "augmentation": aug_type,
                    "scene": scene,
                    "size_bin": size_bin,
                })

    # Aggregate by_aug
    by_aug_summary = {}
    for aug_type, v in by_aug.items():
        by_aug_summary[aug_type] = {
            "mean_iou": float(np.mean(v["mean_ious"])),
            "std_iou": float(np.std(v["mean_ious"])) if len(v["mean_ious"]) > 1 else 0.0,
            "survival_050": float(np.mean(v["survival_050"])),
            "survival_070": float(np.mean(v["survival_070"])),
            "count_drifts": v["count_drifts"],
            "n_pairs": v["n_pairs"],
        }

    # Size bin stability (over mask_level)
    size_bin_stats = {"small": [], "medium": [], "large": []}
    for m in mask_level:
        size_bin_stats[m["size_bin"]].append(m["best_iou"])
    size_bins = {}
    for bin_name, ious in size_bin_stats.items():
        size_bins[bin_name] = {
            "mean_iou": float(np.mean(ious)) if ious else 0.0,
            "survival_050": float(np.mean(np.array(ious) >= 0.5)) if ious else 0.0,
            "count": len(ious),
            "percentile_lo": p33 if bin_name == "small" else (p66 if bin_name == "medium" else None),
            "percentile_hi": p66 if bin_name == "medium" else (None if bin_name == "large" else p33),
        }
    size_bins["_percentiles"] = {"p33": p33, "p66": p66}

    # Oversegmentation summary (original set only)
    oversegmentation = {
        "mask_count_per_image": mask_count_per_image,
        "mask_areas_flat": mask_areas_flat,
        "mean_masks_per_image": float(np.mean(mask_count_per_image)),
        "median_masks_per_image": float(np.median(mask_count_per_image)),
        "median_area": float(np.median(mask_areas_flat)),
        "n_images": len(mask_count_per_image),
    }

    # Per-scene (aggregate over pairs that belong to that scene)
    scene_pairs: dict[str, list[dict]] = {}
    for pair in per_pair:
        scene = scene_class_from_stem(pair["original_stem"])
        if scene not in scene_pairs:
            scene_pairs[scene] = []
        scene_pairs[scene].append(pair)

    per_scene = {}
    for scene, pairs in scene_pairs.items():
        per_scene[scene] = {
            "mean_iou": float(np.mean([p["mean_best_iou"] for p in pairs])),
            "survival_050": float(np.mean([p["survival_rate_050"] for p in pairs])),
            "n_pairs": len(pairs),
        }

    # Per-scene × augmentation (for grouped bar)
    per_scene_aug: dict[str, dict[str, dict]] = {}
    for pair in per_pair:
        scene = scene_class_from_stem(pair["original_stem"])
        aug_type = pair["augmentation"]
        if scene not in per_scene_aug:
            per_scene_aug[scene] = {}
        if aug_type not in per_scene_aug[scene]:
            per_scene_aug[scene][aug_type] = {"mean_ious": [], "survival_050": []}
        per_scene_aug[scene][aug_type]["mean_ious"].append(pair["mean_best_iou"])
        per_scene_aug[scene][aug_type]["survival_050"].append(pair["survival_rate_050"])

    per_scene_aug_summary = {}
    for scene, augs in per_scene_aug.items():
        per_scene_aug_summary[scene] = {}
        for aug_type, v in augs.items():
            per_scene_aug_summary[scene][aug_type] = {
                "mean_iou": float(np.mean(v["mean_ious"])),
                "survival_050": float(np.mean(v["survival_050"])),
                "n": len(v["mean_ious"]),
            }

    analysis = {
        "by_aug_type": by_aug_summary,
        "mask_count_drift": count_drifts,
        "size_bins": size_bins,
        "oversegmentation": oversegmentation,
        "per_scene": per_scene,
        "per_scene_aug": per_scene_aug_summary,
        "mask_level": mask_level,
    }

    analysis_output_path = Path(analysis_output_path)
    analysis_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Analysis saved to {analysis_output_path}")
    return analysis


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze consistency results for reporting and plotting.")
    ap.add_argument("--results", type=Path, default=Path("sam_output/consistency_results.json"), help="consistency_results.json from evaluate_consistency.py")
    ap.add_argument("--original-metadata", type=Path, default=Path("sam_output/debug/metadata"), help="Original SAM metadata dir")
    ap.add_argument("--output", type=Path, default=Path("sam_output/analysis.json"), help="Output analysis JSON")
    ap.add_argument("--size-percentiles", type=float, nargs=2, default=[33.0, 66.0], metavar=("P33", "P66"), help="Percentiles for small/medium/large bins")
    args = ap.parse_args()
    run(args.results, args.original_metadata, args.output, tuple(args.size_percentiles))


if __name__ == "__main__":
    main()
