#!/usr/bin/env python3
"""
Generate augmented versions of images for SAM consistency evaluation.

Creates 1 (or 2) augmentations per image: small rotation, horizontal flip,
slight resize, or mild blur. Saves augmented images and a manifest JSON so
evaluate_consistency.py can inverse-warp masks and compute metrics.

CPU-friendly: by default only 1 augmentation per image (12 extra images = 1x SAM run time).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Augmentation types we support (with invertible geometry for consistency eval)
AUG_TYPES = ["hflip", "rotation", "resize", "blur"]


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path)


def augment_hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1])


def augment_rotation(img: np.ndarray, angle_deg: float = 8.0) -> np.ndarray:
    from scipy.ndimage import rotate
    # angle positive = counter-clockwise; clip to avoid black borders dominating
    out = rotate(img, angle_deg, order=1, reshape=False, mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_resize(img: np.ndarray, scale: float = 0.92) -> np.ndarray:
    h, w = img.shape[:2]
    nw, nh = int(w * scale), int(h * scale)
    pil = Image.fromarray(img)
    pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    out = np.array(pil)
    # Pad back to original size so SAM runs on same dimensions
    if out.shape[0] != h or out.shape[1] != w:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: out.shape[0], : out.shape[1]] = out
        out = canvas
    return out


def augment_blur(img: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    out = gaussian_filter(img.astype(float), sigma=(sigma, sigma, 0), mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def run(
    image_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    augs_per_image: int = 1,
    seed: int = 42,
) -> None:
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if not image_paths:
        raise SystemExit(f"No images in {image_dir}")

    rng = np.random.default_rng(seed)
    manifest = []

    for i, path in enumerate(image_paths):
        stem = path.stem
        img = load_image(path)
        h, w = img.shape[:2]

        # Even distribution: round-robin over AUG_TYPES (e.g. 4 types, 12 images → 3 each)
        for k in range(augs_per_image):
            aug_type = AUG_TYPES[(i * augs_per_image + k) % len(AUG_TYPES)]
            if aug_type == "hflip":
                out_img = augment_hflip(img)
                params = {}
                out_stem = f"{stem}_hflip"
            elif aug_type == "rotation":
                angle = float(rng.uniform(-10, 10))
                out_img = augment_rotation(img, angle)
                params = {"angle_deg": angle}
                out_stem = f"{stem}_rot{angle:.1f}".replace(".", "p").replace("-", "m")
            elif aug_type == "resize":
                scale = float(rng.uniform(0.88, 0.96))
                out_img = augment_resize(img, scale)
                params = {"scale": scale}
                out_stem = f"{stem}_resize{scale:.2f}".replace(".", "p")
            else:  # blur
                sigma = float(rng.uniform(0.8, 1.5))
                out_img = augment_blur(img, sigma)
                params = {"sigma": sigma}
                out_stem = f"{stem}_blur{sigma:.2f}".replace(".", "p")

            out_name = out_stem + path.suffix
            out_path = output_dir / out_name
            save_image(out_img, out_path)

            manifest.append({
                "original_path": path.name,
                "original_stem": stem,
                "augmented_path": out_name,
                "augmented_stem": out_stem,
                "augmentation": aug_type,
                "params": params,
                "original_size": [h, w],
            })

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} augmented images in {output_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate augmented images + manifest for consistency eval.")
    ap.add_argument("--images", type=Path, default=Path("images/mit_indoor_debug"), help="Input image directory")
    ap.add_argument("--output-dir", type=Path, default=Path("images/mit_indoor_debug_aug"), help="Where to save augmented images")
    ap.add_argument("--manifest", type=Path, default=Path("sam_output/aug_manifest.json"), help="Output manifest JSON path")
    ap.add_argument("--augs-per-image", type=int, default=1, choices=(1, 2), help="1 or 2 augmentations per image (default 1 for CPU)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.images, args.output_dir, args.manifest, args.augs_per_image, args.seed)


if __name__ == "__main__":
    main()
