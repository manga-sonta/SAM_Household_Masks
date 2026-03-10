#!/usr/bin/env python3
"""
Download MIT Indoor Scene dataset (indoorCVPR_09), sample 50–100 images
equally from 3 classes (bedroom, kitchen, livingroom), and optionally run
the SAM pipeline on the subset.

Dataset: https://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# 3 classes only (bedroom, kitchen, livingroom)
MIT_INDOOR_CATEGORIES = ["bedroom", "kitchen", "livingroom"]
MIT_INDOOR_URL = "https://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
DEFAULT_TAR = "indoorCVPR_09.tar"
DEFAULT_EXTRACT_DIR = "mit_indoor"
# After extract: mit_indoor/Images/<category>/*.jpg


def download_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, 100.0 * downloaded / total_size)
    sys.stdout.write(f"\rDownloading: {pct:.1f}%")
    sys.stdout.flush()


def download_and_extract(
    work_dir: Path,
    tar_name: str = DEFAULT_TAR,
    extract_dir: str = DEFAULT_EXTRACT_DIR,
    force: bool = False,
) -> Path:
    """Download indoorCVPR_09.tar and extract. Returns path to mit_indoor root."""
    work_dir = Path(work_dir)
    extract_path = work_dir / extract_dir
    tar_path = work_dir / tar_name

    if extract_path.is_dir() and not force:
        print(f"Extract dir already exists: {extract_path} (use --force to re-download/extract)")
        return extract_path

    if not tar_path.is_file() or force:
        print(f"Downloading {MIT_INDOOR_URL} ...")
        urllib.request.urlretrieve(MIT_INDOOR_URL, tar_path, download_progress)
        print()
    else:
        print(f"Using existing tar: {tar_path}")

    if extract_path.is_dir():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True)
    print(f"Extracting to {extract_path} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(extract_path)
    print("Extracted.")
    # Tar may have one top-level dir (e.g. indoorCVPR_09/) or put Images at root
    images_dir = extract_path / "Images"
    if not images_dir.is_dir():
        subdirs = [d for d in extract_path.iterdir() if d.is_dir()]
        for d in subdirs:
            if (d / "Images").is_dir():
                return d
    return extract_path


def sample_and_copy(
    mit_root: Path,
    output_dir: Path,
    categories: list[str],
    total_target: int,
    seed: int = 42,
) -> int:
    """Sample images equally across categories and copy to output_dir. Returns count copied."""
    src_root = mit_root / "Images"
    if not src_root.is_dir():
        raise FileNotFoundError(f"Expected Images dir under {mit_root}: {src_root}")

    random.seed(seed)
    per_class = max(1, total_target // len(categories))
    picked: list[tuple[str, str]] = []  # (category, full_path)

    for cat in categories:
        pattern = os.path.join(src_root, cat, "*.jpg")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"No images for category: {cat}")
            continue
        n = min(per_class, len(paths))
        chosen = random.sample(paths, n)
        for p in chosen:
            picked.append((cat, p))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cat, src in picked:
        fname = os.path.basename(src)
        dst = output_dir / f"{cat}__{fname}"
        shutil.copy2(src, dst)

    print(f"Copied {len(picked)} images to {output_dir}")
    return len(picked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MIT Indoor dataset, sample 50–100 images from 3 classes, optionally run SAM."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for tar and extracted dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images") / "mit_indoor_subset",
        help="Where to copy sampled images (and where SAM will read from if --run-sam)",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=90,
        metavar="N",
        help="Total images to sample (50–100 recommended), equally across 3 classes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download tar and re-extract",
    )
    parser.add_argument(
        "--run-sam",
        action="store_true",
        help="Run batch_sam_masks.py on the sampled images after copying",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="SAM checkpoint path (for --run-sam)",
    )
    parser.add_argument(
        "--sam-output",
        type=Path,
        default=Path("sam_output") / "mit_indoor",
        help="SAM output directory (for --run-sam)",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract
    mit_root = download_and_extract(work_dir, force=args.force)

    # Sample 50–100 (or whatever --total) from 3 classes
    total = max(3, min(args.total, 999))
    n = sample_and_copy(
        mit_root,
        args.output_dir,
        categories=MIT_INDOOR_CATEGORIES,
        total_target=total,
    )
    if n == 0:
        print("No images copied. Exiting.")
        sys.exit(1)

    if not args.run_sam:
        print(f"Done. Run SAM on these images with:")
        print(f"  python batch_sam_masks.py --images {args.output_dir} --output {args.sam_output}")
        return

    # Run SAM pipeline
    script_dir = Path(__file__).resolve().parent
    batch_sam = script_dir / "batch_sam_masks.py"
    if not batch_sam.is_file():
        print(f"batch_sam_masks.py not found at {batch_sam}", file=sys.stderr)
        sys.exit(1)
    cmd = [
        sys.executable,
        str(batch_sam),
        "--images",
        str(args.output_dir),
        "--output",
        str(args.sam_output),
    ]
    if args.sam_checkpoint is not None:
        cmd.extend(["--checkpoint", str(args.sam_checkpoint)])
    print("Running SAM pipeline:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
