#!/usr/bin/env python3
"""
Run SAM 3D Objects on SAM-masked images to get 3D reconstructions (one PLY per object).

Requires the [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) repo to be
cloned and its environment installed (see doc/setup.md). Run this script with that
environment activated.

Usage:
  # From sam-household-masks with sam3d-objects env activated:
  python run_sam3d.py --sam3d-repo /path/to/sam-3d-objects \\
    --images images/mit_indoor_debug --masks-root sam_output/debug/masks \\
    --output sam3d_output

Inputs:
  - --images: directory with original RGB images (e.g. bedroom__375.jpg).
  - --masks-root: directory containing per-image mask folders, e.g. masks_root/bedroom__375/mask_0000.png ...

Outputs:
  - One PLY per object: output/<stem>/object_0000.ply, object_0001.ply, ...
  - PLY files are 3D Gaussian Splatting format (see README for conversion to mesh / Isaac Sim).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _ensure_sam3d_import(sam3d_repo: Path):
    """Prepend sam-3d-objects repo to path and ensure notebook.inference is importable."""
    repo = Path(sam3d_repo).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from notebook.inference import Inference  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Could not import notebook.inference from sam-3d-objects. "
            "Clone https://github.com/facebookresearch/sam-3d-objects, install its env (see doc/setup.md), "
            "and run this script with that env activated, e.g.:\n"
            "  mamba activate sam3d-objects\n"
            "  python run_sam3d.py --sam3d-repo /path/to/sam-3d-objects ..."
        ) from e


def load_image_numpy(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def load_mask_numpy(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    mask = (arr > 0) if arr.ndim == 2 else (arr[..., 0] > 0)
    return mask.astype(bool)


def run(
    sam3d_repo: Path,
    images_dir: Path,
    masks_root: Path,
    output_dir: Path,
    config_name: str = "hf",
    seed: int = 42,
    max_masks_per_image: int | None = None,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> None:
    _ensure_sam3d_import(sam3d_repo)
    from notebook.inference import Inference

    repo = Path(sam3d_repo).resolve()
    config_path = repo / "checkpoints" / config_name / "pipeline.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"SAM 3D config not found: {config_path}. "
            "Download checkpoints (see https://github.com/facebookresearch/sam-3d-objects doc/setup.md)."
        )

    inference = Inference(str(config_path), compile=False)
    images_dir = Path(images_dir)
    masks_root = Path(masks_root)
    output_dir = Path(output_dir)

    # List stems: subdirs of masks_root that contain mask_*.png
    stems = sorted(
        d.name for d in masks_root.iterdir()
        if d.is_dir() and any(d.glob("mask_*.png"))
    )
    if not stems:
        raise SystemExit(f"No mask folders found under {masks_root}")

    for stem in stems:
        # Resolve image path (stem may not include extension)
        image_path = None
        for ext in image_extensions:
            p = images_dir / f"{stem}{ext}"
            if p.is_file():
                image_path = p
                break
        if image_path is None:
            cands = list(images_dir.glob(f"{stem}.*"))
            if cands:
                image_path = cands[0]
        if not image_path or not image_path.is_file():
            print(f"Skip {stem}: no image found in {images_dir}")
            continue

        mask_dir = masks_root / stem
        mask_paths = sorted(mask_dir.glob("mask_*.png"))
        if max_masks_per_image is not None:
            mask_paths = mask_paths[: max_masks_per_image]

        if not mask_paths:
            print(f"Skip {stem}: no masks in {mask_dir}")
            continue

        image = load_image_numpy(image_path)
        out_stem_dir = output_dir / stem
        out_stem_dir.mkdir(parents=True, exist_ok=True)

        for mp in mask_paths:
            # object_0000.ply from mask_0000.png
            idx = mp.stem.replace("mask_", "")
            out_ply = out_stem_dir / f"object_{idx}.ply"
            if out_ply.is_file():
                print(f"Exists: {out_ply}")
                continue

            mask = load_mask_numpy(mp)
            try:
                output = inference(image, mask, seed=seed)
            except Exception as e:
                print(f"Error {stem} mask {mp.name}: {e}")
                continue

            gs = output.get("gs")
            if gs is None:
                print(f"No 'gs' in output for {stem} {mp.name}")
                continue
            gs.save_ply(str(out_ply))
            print(f"Saved: {out_ply}")

    print(f"Done. Outputs under {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SAM 3D Objects on SAM mask outputs; export one PLY per object."
    )
    ap.add_argument(
        "--sam3d-repo",
        type=Path,
        required=True,
        help="Path to cloned sam-3d-objects repo (https://github.com/facebookresearch/sam-3d-objects)",
    )
    ap.add_argument(
        "--images",
        type=Path,
        default=Path("images/mit_indoor_debug"),
        help="Directory with original RGB images (names must match mask folder stems)",
    )
    ap.add_argument(
        "--masks-root",
        type=Path,
        default=Path("sam_output/debug/masks"),
        help="Directory containing per-image mask folders (e.g. masks_root/bedroom__375/mask_0000.png)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("sam3d_output"),
        help="Output directory for PLY files (output/<stem>/object_0000.ply, ...)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="hf",
        help="Checkpoint config name under checkpoints/ (default: hf)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-masks-per-image",
        type=int,
        default=None,
        help="Limit number of objects per image (e.g. 5 for quick test)",
    )
    args = ap.parse_args()
    run(
        args.sam3d_repo,
        args.images,
        args.masks_root,
        args.output,
        config_name=args.config,
        seed=args.seed,
        max_masks_per_image=args.max_masks_per_image,
    )


if __name__ == "__main__":
    main()
