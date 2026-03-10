#!/usr/bin/env python3
"""
Batch SAM (Segment Anything Model) mask generation for household images.

Uses SAM's automatic mask generation (SamAutomaticMaskGenerator): every relevant
object in each image is segmented automatically — no bounding boxes, clicks, or
other user input required. A grid of point prompts is run over the image and
masks are filtered by quality and NMS.

Processes a folder of images, extracts instance masks and metadata per image,
and saves:
  - masks/  : one PNG per mask (or per-image combined) for downstream 3D (e.g. SAM3D)
  - metadata/: JSON per image with bbox, area, predicted_iou, stability_score, etc.
  - viz/    : optional overlay visualizations

Usage:
  python batch_sam_masks.py --images ./my_household_photos --output ./sam_output [--checkpoint path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

# SAM imports (requires segment-anything from GitHub)
try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
except ImportError as e:
    print("Install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git", file=sys.stderr)
    raise SystemExit(1) from e

# -----------------------------------------------------------------------------
# Model registry: map name -> (build_fn, default_checkpoint_filename)
# Checkpoints: download from https://github.com/facebookresearch/segment-anything#model-checkpoints
# e.g. https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# -----------------------------------------------------------------------------
SAM_MODELS = {
    "vit_b": (build_sam_vit_b, "sam_vit_b_01ec64.pth"),
    "vit_l": (build_sam_vit_l, "sam_vit_l_0b3195.pth"),
    "vit_h": (build_sam_vit_h, "sam_vit_h_4b8939.pth"),
}


def _to_serializable(obj: Any) -> Any:
    """Convert numpy types to native Python for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return obj


def load_image(path: Path) -> np.ndarray:
    """Load image as HWC uint8 RGB."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def build_mask_generator(
    checkpoint: str | Path,
    model_type: str = "vit_b",
    device: str | None = None,
    points_per_side: int | None = 32,
    points_per_batch: int = 64,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    crop_n_layers: int = 1,
    min_mask_region_area: int = 100,
) -> SamAutomaticMaskGenerator:
    """Build SAM model and automatic mask generator."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type not in SAM_MODELS:
        raise ValueError(f"model_type must be one of {list(SAM_MODELS)}")
    build_fn, _ = SAM_MODELS[model_type]
    sam = build_fn(checkpoint=str(checkpoint))
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
        output_mode="binary_mask",
    )
    return mask_generator


def masks_to_metadata_records(masks: list[dict]) -> list[dict]:
    """Convert SAM mask list to JSON-serializable metadata (no raw segmentation array)."""
    records = []
    for m in masks:
        rec = {
            "bbox": m["bbox"],           # [x, y, w, h]
            "area": m["area"],
            "predicted_iou": m["predicted_iou"],
            "stability_score": m["stability_score"],
            "point_coords": m["point_coords"],
            "crop_box": m["crop_box"],
        }
        records.append(_to_serializable(rec))
    return records


def run_batch(
    image_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    model_type: str = "vit_b",
    save_individual_masks: bool = True,
    save_visualization: bool = True,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    **sam_kwargs: Any,
) -> None:
    """Run SAM on all images in image_dir and write masks + metadata under output_dir."""
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    meta_dir = output_dir / "metadata"
    viz_dir = output_dir / "viz"
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if save_visualization:
        viz_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    if not image_paths:
        print(f"No images found in {image_dir} with extensions {image_extensions}")
        return

    mask_generator = build_mask_generator(checkpoint, model_type=model_type, **sam_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for path in tqdm(sorted(image_paths), desc="Images"):
        stem = path.stem
        try:
            image = load_image(path)
        except Exception as e:
            tqdm.write(f"Skip {path}: failed to load — {e}")
            continue

        with torch.inference_mode():
            masks = mask_generator.generate(image)

        # Sort by area descending so "main" objects tend to have lower index
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        # Metadata (serializable, no big arrays)
        meta = {
            "image": path.name,
            "image_size": [int(image.shape[0]), int(image.shape[1])],
            "num_masks": len(masks),
            "masks": masks_to_metadata_records(masks),
        }
        with open(meta_dir / f"{stem}.json", "w") as f:
            json.dump(_to_serializable(meta), f, indent=2)

        # Save each mask as PNG (binary 0/255) for downstream 3D pipeline
        if save_individual_masks:
            per_image_dir = masks_dir / stem
            per_image_dir.mkdir(parents=True, exist_ok=True)
            for i, m in enumerate(masks):
                seg = m["segmentation"]
                mask_uint8 = (seg.astype(np.uint8)) * 255
                Image.fromarray(mask_uint8).save(per_image_dir / f"mask_{i:04d}.png")

        # Optional: overlay visualization
        if save_visualization and masks:
            import cv2
            overlay = image.copy()
            np.random.seed(42)
            for m in masks:
                color = np.random.randint(0, 255, 3, dtype=np.uint8)
                overlay[m["segmentation"]] = overlay[m["segmentation"]] * 0.5 + color * 0.5
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            cv2.imwrite(str(viz_dir / f"{stem}.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Done. Processed {len(image_paths)} images. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch SAM mask generation for household images (masks + metadata for 3D pipeline). "
        "Uses automatic mask generation: no clicks or bounding boxes required."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sam_output"),
        help="Output directory (masks/, metadata/, viz/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SAM checkpoint .pth (default: look for sam_vit_b_01ec64.pth in cwd)",
    )
    parser.add_argument(
        "--model",
        choices=list(SAM_MODELS),
        default="vit_b",
        help="SAM backbone: vit_b (fast), vit_l, vit_h (best quality)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable saving overlay visualizations",
    )
    parser.add_argument(
        "--no-individual-masks",
        action="store_true",
        help="Do not save per-mask PNGs (only metadata)",
    )
    # SAM tuning (household scenes: more points can help for cluttered scenes)
    parser.add_argument("--points-per-side", type=int, default=32, help="Grid density for auto prompts")
    parser.add_argument("--points-per-batch", type=int, default=64, help="Points per batch (GPU memory)")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88, help="Mask quality threshold")
    parser.add_argument("--stability-score-thresh", type=float, default=0.95, help="Stability threshold")
    parser.add_argument("--crop-n-layers", type=int, default=1, help="Extra crop layers for small objects")
    parser.add_argument("--min-mask-area", type=int, default=100, help="Remove tiny mask regions")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        _, default_name = SAM_MODELS[args.model]
        # Check cwd and script dir
        for d in (Path.cwd(), Path(__file__).resolve().parent):
            c = d / default_name
            if c.is_file():
                checkpoint = c
                break
        if checkpoint is None:
            print(
                f"Error: no checkpoint found. Download {SAM_MODELS[args.model][1]} from\n"
                "  https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                "  and pass --checkpoint /path/to/sam_vit_b_01ec64.pth",
                file=sys.stderr,
            )
            sys.exit(1)

    if not args.images.is_dir():
        print(f"Error: --images must be a directory: {args.images}", file=sys.stderr)
        sys.exit(1)

    run_batch(
        image_dir=args.images,
        output_dir=args.output,
        checkpoint=checkpoint,
        model_type=args.model,
        save_individual_masks=not args.no_individual_masks,
        save_visualization=not args.no_viz,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_area,
    )


if __name__ == "__main__":
    main()
