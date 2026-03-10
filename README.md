# SAM batch mask generation for household scenes

Batch [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) inference on household images (beds, chairs, desks, paintings, lamps, etc.) to produce **masks** and **metadata** for a downstream pipeline: 2D → 3D assets (e.g. SAM3D) → Isaac Sim → robot training (e.g. opening drawers).

**Automatic segmentation:** The pipeline uses SAM’s **automatic mask generation** (`SamAutomaticMaskGenerator`). Every relevant object in each image is segmented automatically — no bounding boxes, clicks, or other user input required. A grid of point prompts is run over the image; masks are filtered by quality and NMS.

## Pipeline overview

1. **This repo**: SAM on batches of images → per-image instance masks + JSON metadata (bbox, area, predicted_iou, stability_score).
2. **Next**: Use masks + metadata for 3D reconstruction / asset generation (e.g. depth + mesh, or SAM3D / SA3D).
3. **Then**: Import 3D scenes into **Isaac Sim**, add a robot, and train policies (e.g. manipulation).

## Setup

### 1. Environment

```bash
cd sam-household-masks
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

- **GPU**: Install a CUDA-enabled PyTorch if you have a GPU (faster): see [pytorch.org](https://pytorch.org).
- SAM is installed from the official GitHub repo via `requirements.txt`.

### 2. Download SAM checkpoint

Pick one (larger = better quality, slower):

| Model  | Checkpoint            | Size   | Use case        |
|--------|------------------------|--------|------------------|
| `vit_b` | `sam_vit_b_01ec64.pth` | ~375 MB | Fast, good default |
| `vit_l` | `sam_vit_l_0b3195.pth` | ~1.2 GB | Better quality   |
| `vit_h` | `sam_vit_h_4b8939.pth` | ~2.5 GB | Best quality     |

Download from [segment-anything model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) (e.g. official links or [Hugging Face](https://huggingface.co/ybelkada/segment-anything/tree/main/checkpoints)) and place in this directory, or pass `--checkpoint /path/to/sam_vit_b_01ec64.pth`.

Example (Vit-B):

```bash
wget -O sam_vit_b_01ec64.pth "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
```

## Run on Google Colab (100–150 images)

To run the full pipeline (SAM → augmentation → consistency evaluation → plots) on 100–150 images with a **GPU**:

1. **Push this repo to GitHub** (create a new repo and `git push` your code).
2. Open [Google Colab](https://colab.research.google.com), **File → Upload notebook** and upload `colab_run_pipeline.ipynb` from this repo (or copy its cells into a new notebook).
3. In the first cell, set **REPO_URL** to your repo’s clone URL (e.g. `https://github.com/YOUR_USER/sam-household-masks.git`). Leave empty if you upload the project folder to Colab instead.
4. Set **TOTAL_IMAGES** (e.g. 120) in the MIT Indoor cell.
5. **Runtime → Change runtime type → GPU**, then run all cells.

The notebook will: clone repo → install deps → download SAM checkpoint → download MIT Indoor and sample images → run SAM on originals → generate **even** augmentations (hflip / rotation / resize / blur) → run SAM on augmented → evaluate → analyze → plot. Finally you can download a zip of `sam_output/` and images.

Pipeline stops at **plots** (no SAM3D).

## MIT Indoor dataset (bedroom, kitchen, livingroom)

Download the [MIT Indoor Scene](https://groups.csail.mit.edu/vision/LabelMe/NewImages/) dataset, sample **50–100 images** equally from **3 classes** (bedroom, kitchen, livingroom), and optionally run SAM in one go:

```bash
# Download, extract, sample 90 images (30 per class), copy to images/mit_indoor_subset
python download_mit_indoor.py --total 90

# Same, then run SAM on the subset (requires SAM checkpoint)
python download_mit_indoor.py --total 90 --run-sam --sam-checkpoint ./sam_vit_b_01ec64.pth
```

Options: `--work-dir` (where to put the tar and extracted data), `--output-dir` (where sampled images are copied), `--total` (e.g. 60–99), `--force` (re-download and re-extract), `--run-sam`, `--sam-checkpoint`, `--sam-output`.

## Usage

```bash
# Default: read images from ./images, write to ./sam_output, use vit_b (fully automatic, no clicks)
python batch_sam_masks.py --images ./images --output ./sam_output

# Specify checkpoint and larger model
python batch_sam_masks.py --images ./my_photos --output ./out --checkpoint ./sam_vit_h_4b8939.pth --model vit_h

# Tune for cluttered household scenes (more segments, small objects)
python batch_sam_masks.py --images ./images --output ./out --points-per-side 40 --crop-n-layers 2
```

### Output layout

```
sam_output/
├── masks/
│   └── <image_stem>/
│       ├── mask_0000.png   # binary mask per instance
│       ├── mask_0001.png
│       └── ...
├── metadata/
│   └── <image_stem>.json   # bbox, area, predicted_iou, stability_score, etc.
└── viz/
    └── <image_stem>.png    # optional overlay visualization
```

### Metadata JSON (per image)

- `image`, `image_size`, `num_masks`
- `masks`: list of `{ "bbox", "area", "predicted_iou", "stability_score", "point_coords", "crop_box" }` — suitable for filtering by quality or size before 3D.

### Options

- `--no-viz`: skip overlay images.
- `--no-individual-masks`: only write metadata (no per-mask PNGs).
- `--points-per-side`, `--points-per-batch`, `--pred-iou-thresh`, `--stability-score-thresh`, `--crop-n-layers`, `--min-mask-area`: tune SAM for your resolution and clutter.

## Quantitative consistency evaluation (augmentations + metrics)

Pipeline to measure how stable SAM masks are under mild augmentations (rotation, flip, resize, blur): generate augmented images, run SAM on them, then compare masks (after inverse-warping) with the original run.

**Default: 1 augmentation per image** so one extra SAM run stays manageable on CPU (~same time as your 12-image run).

### 1. Generate augmented images and manifest

From the repo root (e.g. `Desktop/sam-household-masks`):

```bash
python generate_augmented.py --images images/mit_indoor_debug --output-dir images/mit_indoor_debug_aug --manifest sam_output/aug_manifest.json --augs-per-image 1
```

- Reads `images/mit_indoor_debug` (your 12 images).
- Writes 12 augmented images into `images/mit_indoor_debug_aug` with **even** augmentation types: round-robin over hflip, rotation, resize, blur (e.g. 12 images → 3 of each).
- Saves `sam_output/aug_manifest.json` (mapping original ↔ augmented and augmentation type/params for inverse-warping).

### 2. Run SAM on the augmented images

```bash
python batch_sam_masks.py --images images/mit_indoor_debug_aug --output sam_output/debug_aug --checkpoint ./sam_vit_b_01ec64.pth
```

(Adjust `--checkpoint` if your checkpoint is elsewhere.) This will take about as long as your original 12-image run (~4 h on CPU).

### 3. Evaluate consistency

```bash
python evaluate_consistency.py --original-output sam_output/debug --augmented-output sam_output/debug_aug --manifest sam_output/aug_manifest.json --results sam_output/consistency_results.json
```

- Loads original masks from `sam_output/debug/masks/`, augmented masks from `sam_output/debug_aug/masks/`.
- For each (original, augmented) pair: inverse-warps augmented masks to original image space, then for each original mask computes the best IoU with any warped augmented mask.
- Prints a summary and writes `sam_output/consistency_results.json`.

**Metrics:**

- **Mean / median best IoU**: per original mask, max IoU with any augmented mask (after alignment); averaged over masks and/or images.
- **Survival @ 0.5 (and 0.7)**: fraction of original masks that have a match ≥ 0.5 (or 0.7) IoU.
- **Per-pair**: same metrics per (original image, augmentation type), plus original vs augmented mask counts.

To use 2 augmentations per image (24 augmented images, ~2× SAM time), run step 1 with `--augs-per-image 2` and ensure step 2 writes to the same `sam_output/debug_aug` (it will contain 24 stems). Step 3 stays the same.

### 4. Analyze for report (group by aug, drift, size bins, oversegmentation, per-scene)

```bash
python analyze_consistency.py --results sam_output/consistency_results.json --original-metadata sam_output/debug/metadata --output sam_output/analysis.json
```

- Groups results **by augmentation type** (mean IoU, survival@0.5, mask count drift).
- Computes **mask count drift** (augmented − original) per pair and by aug type.
- Computes **small / medium / large object stability** (masks binned by area percentiles; mean IoU and survival per bin).
- **Oversegmentation stats** on the original set: mask count per image, distribution of mask areas.
- **Per-scene-class** (bedroom, kitchen, livingroom from stem prefix): mean IoU and survival; per-scene × augmentation for grouped comparison.
- Writes `sam_output/analysis.json` (used by the plotting script).

### 5. Generate plots for report

```bash
python plot_consistency.py --analysis sam_output/analysis.json --output-dir sam_output/plots
```

Produces in `sam_output/plots/`:

| Figure | Description |
|--------|-------------|
| `bar_mean_iou_by_aug.png` | Bar plot: mean best IoU by augmentation type |
| `bar_survival_050_by_aug.png` | Bar plot: survival@0.5 by augmentation type |
| `box_mask_count_drift.png` | Box plot: mask count drift by augmentation type |
| `hist_mask_sizes.png` | Histogram: mask area distribution (original set) |
| `scatter_area_vs_iou.png` | Scatter: original mask area vs best IoU (colored by aug) |
| `bar_mean_iou_by_scene.png` | Bar: mean IoU by scene class (bedroom / kitchen / livingroom) |
| `bar_scene_aug_grouped.png` | Grouped bar: mean IoU by scene and augmentation type |
| `bar_mean_iou_by_size_bin.png` | Bar: mean IoU by object size bin (small / medium / large) |

## SAM 3D Objects (masks → 3D)

[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) turns each masked object into a **3D reconstruction** (one **PLY** file per object). The PLY files are **3D Gaussian Splatting** format, not triangle meshes.

### Prerequisites

- Clone and install [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) (see their [doc/setup.md](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)): Linux, NVIDIA GPU (≥32 GB VRAM recommended), conda/mamba env.
- Download checkpoints (Hugging Face; access may require approval).

### Run SAM 3D on our masks

From this repo, with the **sam3d-objects** environment activated:

```bash
python run_sam3d.py --sam3d-repo /path/to/sam-3d-objects \
  --images images/mit_indoor_debug \
  --masks-root sam_output/debug/masks \
  --output sam3d_output
```

- Reads original images from `--images` and masks from `--masks-root/<stem>/mask_*.png`.
- Writes one PLY per object: `sam3d_output/<stem>/object_0000.ply`, `object_0001.ply`, ...
- Use `--max-masks-per-image N` to limit objects per image (e.g. for a quick test).

### What SAM 3D outputs

- **Format:** **PLY** = 3D Gaussian Splatting (positions, scale, rotation, opacity, color). Not OBJ/mesh by default.
- **Next steps for Isaac Sim:** Isaac Sim expects USD and usually mesh-based or rigid-body assets. So: convert splat PLY → mesh (OBJ/FBX) if needed, then convert to USD (e.g. via Isaac Lab’s MeshConverter), add physics/articulation, then add the robot. See [doc/SAM3D_AND_ISAAC_SIM.md](doc/SAM3D_AND_ISAAC_SIM.md) for a short guide.

## Next steps (3D and simulation)

- **2D → 3D**: Use depth estimation (e.g. DPT, MiDaS) + back-projection, or multi-view reconstruction, or a 3D-from-masks method (e.g. SAM3D / SA3D) to get meshes or point clouds per object.
- **Isaac Sim**: Import 3D assets, build a scene, add rigid bodies and joints for drawers/doors, then use reinforcement learning or imitation learning for tasks like opening drawers.

This script only does the SAM mask + metadata step; the rest of the pipeline is in your 3D and sim stack.
