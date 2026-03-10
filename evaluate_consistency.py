#!/usr/bin/env python3
"""
Evaluate mask consistency between original and augmented SAM runs.

Reads original SAM output (masks + metadata), augmented SAM output, and the
augmentation manifest. Inverse-warps augmented masks to original image space,
then computes:
  - Per-mask best IoU: for each original mask, max IoU with any augmented mask (after alignment)
  - Mean / median best IoU per image and overall
  - Mask survival rate: fraction of original masks with best IoU >= 0.5
  - Count consistency: original vs augmented mask counts

Run after: (1) SAM on original images, (2) generate_augmented.py, (3) SAM on augmented images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return (arr > 0).astype(np.uint8)


def load_all_masks(masks_dir: Path, stem: str) -> list[np.ndarray]:
    folder = masks_dir / stem
    if not folder.is_dir():
        return []
    masks = []
    for p in sorted(folder.glob("mask_*.png")):
        masks.append(load_mask(p))
    return masks


def inverse_warp_mask(
    mask: np.ndarray,
    aug_type: str,
    params: dict,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Warp augmented mask back to original image space (target_h x target_w)."""
    if aug_type == "hflip":
        return np.ascontiguousarray(mask[:, ::-1])
    if aug_type == "rotation":
        from scipy.ndimage import rotate
        angle = params.get("angle_deg", 0.0)
        out = rotate(mask.astype(float), -angle, order=1, reshape=False, mode="constant", cval=0)
        return (out > 0.5).astype(np.uint8)
    if aug_type == "resize":
        scale = params.get("scale", 1.0)
        if scale <= 0 or mask.shape[0] != target_h or mask.shape[1] != target_w:
            return mask
        # Augmented image was (nh, nw) content padded to (target_h, target_w). Crop mask to (nh, nw), then upsample to (target_h, target_w).
        nh, nw = int(target_h * scale), int(target_w * scale)
        cropped = mask[:nh, :nw]
        pil = Image.fromarray((cropped * 255).astype(np.uint8))
        pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
        return (np.array(pil) > 0).astype(np.uint8)
    if aug_type == "blur":
        return mask
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks (same shape)."""
    if a.shape != b.shape:
        return 0.0
    a = a.ravel().astype(bool)
    b = b.ravel().astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def evaluate_one_pair(
    orig_masks: list[np.ndarray],
    aug_masks: list[np.ndarray],
    aug_type: str,
    params: dict,
    original_size: list[int],
) -> dict:
    """For one (original image, augmented image) pair: warp aug masks to original space, compute metrics."""
    target_h, target_w = original_size[0], original_size[1]
    # Warp each augmented mask to original space
    warped = []
    for m in aug_masks:
        if m.shape[0] != target_h or m.shape[1] != target_w:
            from PIL import Image
            pil = Image.fromarray((m * 255).astype(np.uint8))
            pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
            m = (np.array(pil) > 0).astype(np.uint8)
        warped.append(inverse_warp_mask(m, aug_type, params, target_h, target_w))

    if not orig_masks or not warped:
        return {
            "num_original": len(orig_masks),
            "num_augmented": len(aug_masks),
            "mean_best_iou": 0.0,
            "median_best_iou": 0.0,
            "survival_rate_050": 0.0,
            "survival_rate_070": 0.0,
            "best_ious": [],
        }

    best_ious = []
    for om in orig_masks:
        ious = [mask_iou(om, wm) for wm in warped]
        best_ious.append(max(ious) if ious else 0.0)

    best_ious_arr = np.array(best_ious)
    return {
        "num_original": len(orig_masks),
        "num_augmented": len(aug_masks),
        "mean_best_iou": float(np.mean(best_ious_arr)),
        "median_best_iou": float(np.median(best_ious_arr)),
        "survival_rate_050": float(np.mean(best_ious_arr >= 0.5)),
        "survival_rate_070": float(np.mean(best_ious_arr >= 0.7)),
        "best_ious": best_ious_arr.tolist(),
    }


def run(
    original_sam_output: Path,
    augmented_sam_output: Path,
    manifest_path: Path,
    results_path: Path | None = None,
) -> dict:
    original_sam_output = Path(original_sam_output)
    augmented_sam_output = Path(augmented_sam_output)
    manifest_path = Path(manifest_path)

    with open(manifest_path) as f:
        manifest = json.load(f)

    orig_masks_dir = original_sam_output / "masks"
    aug_masks_dir = augmented_sam_output / "masks"

    per_pair = []
    for entry in manifest:
        orig_stem = entry["original_stem"]
        aug_stem = entry["augmented_stem"]
        aug_type = entry["augmentation"]
        params = entry.get("params", {})
        orig_size = entry.get("original_size")

        orig_masks = load_all_masks(orig_masks_dir, orig_stem)
        aug_masks = load_all_masks(aug_masks_dir, aug_stem)

        if orig_size is None:
            if orig_masks:
                orig_size = [orig_masks[0].shape[0], orig_masks[0].shape[1]]
            else:
                orig_size = [480, 640]

        r = evaluate_one_pair(
            orig_masks, aug_masks, aug_type, params, orig_size
        )
        r["original_stem"] = orig_stem
        r["augmented_stem"] = aug_stem
        r["augmentation"] = aug_type
        per_pair.append(r)

    # Aggregate
    mean_ious = [p["mean_best_iou"] for p in per_pair]
    survival_050 = [p["survival_rate_050"] for p in per_pair]
    survival_070 = [p["survival_rate_070"] for p in per_pair]

    summary = {
        "num_pairs": len(per_pair),
        "overall_mean_best_iou": float(np.mean(mean_ious)),
        "overall_median_best_iou": float(np.median(mean_ious)),
        "overall_survival_rate_050": float(np.mean(survival_050)),
        "overall_survival_rate_070": float(np.mean(survival_070)),
        "per_pair": per_pair,
    }

    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {results_path}")

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("SAM MASK CONSISTENCY (original vs augmented)")
    print("=" * 60)
    print(f"  Pairs evaluated:     {summary['num_pairs']}")
    print(f"  Mean best IoU:      {summary['overall_mean_best_iou']:.4f}")
    print(f"  Median best IoU:    {summary['overall_median_best_iou']:.4f}")
    print(f"  Survival @ 0.5:     {summary['overall_survival_rate_050']:.2%} (masks with a match >= 0.5 IoU)")
    print(f"  Survival @ 0.7:     {summary['overall_survival_rate_070']:.2%}")
    print("=" * 60)
    print("\nPer image (original_stem / augmentation):")
    for p in summary["per_pair"]:
        print(f"  {p['original_stem']} | {p['augmentation']:8s} | mean_iou={p['mean_best_iou']:.3f} survival_0.5={p['survival_rate_050']:.2%} n_orig={p['num_original']} n_aug={p['num_augmented']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate mask consistency between original and augmented SAM outputs.")
    ap.add_argument("--original-output", type=Path, default=Path("sam_output/debug"), help="Original SAM output dir (masks/, metadata/)")
    ap.add_argument("--augmented-output", type=Path, default=Path("sam_output/debug_aug"), help="Augmented SAM output dir")
    ap.add_argument("--manifest", type=Path, default=Path("sam_output/aug_manifest.json"), help="Manifest from generate_augmented.py")
    ap.add_argument("--results", type=Path, default=Path("sam_output/consistency_results.json"), help="Where to save JSON results")
    args = ap.parse_args()

    summary = run(
        args.original_output,
        args.augmented_output,
        args.manifest,
        args.results,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
