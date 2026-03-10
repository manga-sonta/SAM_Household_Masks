#!/usr/bin/env python3
"""
Generate report figures from consistency analysis.

Reads analysis.json (from analyze_consistency.py) and produces:
  - Bar: mean IoU by augmentation type
  - Bar: survival@0.5 by augmentation type
  - Box: mask count drift by augmentation type
  - Histogram: mask area distribution (original set)
  - Scatter: original mask area vs best IoU (optionally by aug type)
  - Per-scene: mean IoU by scene class; grouped bar scene × aug type
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_analysis(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def bar_mean_iou_by_aug(analysis: dict, out_path: Path) -> None:
    by_aug = analysis["by_aug_type"]
    aug_types = list(by_aug.keys())
    means = [by_aug[a]["mean_iou"] for a in aug_types]
    stds = [by_aug[a]["std_iou"] for a in aug_types]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(aug_types))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(aug_types)
    ax.set_ylabel("Mean best IoU")
    ax.set_title("Mean best IoU by augmentation type")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def bar_survival_by_aug(analysis: dict, out_path: Path) -> None:
    by_aug = analysis["by_aug_type"]
    aug_types = list(by_aug.keys())
    surv = [by_aug[a]["survival_050"] for a in aug_types]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(aug_types))
    ax.bar(x, surv, color="coral", edgecolor="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(aug_types)
    ax.set_ylabel("Survival @ 0.5")
    ax.set_title("Mask survival rate (IoU ≥ 0.5) by augmentation type")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def box_mask_count_drift(analysis: dict, out_path: Path) -> None:
    by_aug = analysis["by_aug_type"]
    aug_types = list(by_aug.keys())
    data = [by_aug[a]["count_drifts"] for a in aug_types]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=aug_types, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")
        patch.set_alpha(0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Mask count drift (augmented − original)")
    ax.set_title("Mask count drift by augmentation type")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def hist_mask_sizes(analysis: dict, out_path: Path) -> None:
    areas = analysis["oversegmentation"]["mask_areas_flat"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(areas, bins=50, color="teal", alpha=0.8, edgecolor="black")
    ax.set_xlabel("Mask area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of mask sizes (original set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def scatter_area_vs_iou(analysis: dict, out_path: Path) -> None:
    mask_level = analysis["mask_level"]
    areas = np.array([m["area"] for m in mask_level])
    ious = np.array([m["best_iou"] for m in mask_level])
    augs = [m["augmentation"] for m in mask_level]
    fig, ax = plt.subplots(figsize=(6, 5))
    for aug in sorted(set(augs)):
        idx = [i for i, a in enumerate(augs) if a == aug]
        ax.scatter(areas[idx], ious[idx], label=aug, alpha=0.5, s=8)
    ax.set_xlabel("Original mask area (pixels)")
    ax.set_ylabel("Best IoU (after augmentation)")
    ax.set_title("Original mask area vs consistency (best IoU)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def bar_per_scene(analysis: dict, out_path: Path) -> None:
    per_scene = analysis["per_scene"]
    scenes = list(per_scene.keys())
    means = [per_scene[s]["mean_iou"] for s in scenes]
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(scenes))
    ax.bar(x, means, color="mediumpurple", edgecolor="indigo")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Mean best IoU")
    ax.set_title("Mean best IoU by scene class")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def grouped_bar_scene_aug(analysis: dict, out_path: Path) -> None:
    per_scene_aug = analysis["per_scene_aug"]
    scenes = list(per_scene_aug.keys())
    aug_types = sorted(set().union(*(set(per_scene_aug[s].keys()) for s in scenes)))
    x = np.arange(len(scenes))
    width = 0.8 / len(aug_types)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(aug_types)))
    for i, aug in enumerate(aug_types):
        vals = [per_scene_aug[s].get(aug, {}).get("mean_iou", np.nan) for s in scenes]
        off = (i - len(aug_types) / 2 + 0.5) * width
        ax.bar(x + off, vals, width, label=aug, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Mean best IoU")
    ax.set_title("Mean best IoU by scene class and augmentation type")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def bar_size_bins(analysis: dict, out_path: Path) -> None:
    sb = analysis["size_bins"]
    bins_order = ["small", "medium", "large"]
    means = [sb[b]["mean_iou"] for b in bins_order if b in sb and not b.startswith("_")]
    labels = [b for b in bins_order if b in sb and not b.startswith("_")]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#e74c3c", "#f39c12", "#27ae60"], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean best IoU")
    ax.set_title("Mask consistency by object size (small / medium / large)")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot consistency analysis for report.")
    ap.add_argument("--analysis", type=Path, default=Path("sam_output/analysis.json"), help="analysis.json from analyze_consistency.py")
    ap.add_argument("--output-dir", type=Path, default=Path("sam_output/plots"), help="Directory to save figures")
    args = ap.parse_args()

    analysis = load_analysis(args.analysis)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_mean_iou_by_aug(analysis, out_dir / "bar_mean_iou_by_aug.png")
    bar_survival_by_aug(analysis, out_dir / "bar_survival_050_by_aug.png")
    box_mask_count_drift(analysis, out_dir / "box_mask_count_drift.png")
    hist_mask_sizes(analysis, out_dir / "hist_mask_sizes.png")
    scatter_area_vs_iou(analysis, out_dir / "scatter_area_vs_iou.png")
    bar_per_scene(analysis, out_dir / "bar_mean_iou_by_scene.png")
    grouped_bar_scene_aug(analysis, out_dir / "bar_scene_aug_grouped.png")
    bar_size_bins(analysis, out_dir / "bar_mean_iou_by_size_bin.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
