```markdown
# UGRA — Temporal Needle + Anatomy Segmentation (CVAT → ConvLSTM-UNet → “Live” Frame Demo)

This repo implements an end-to-end pipeline to:
1) convert **CVAT 1.1 mask annotations** into per-frame **needle** + **anatomy** masks and temporal manifests  
2) train a **temporal segmentation model** (ConvLSTM-UNet) for:
   - **Needle**: binary segmentation (needle vs background)
   - **Anatomy**: multiclass segmentation (background / nerve / artery / muscle)
3) debug inference visually (side-by-side GT vs Pred)
4) run a **“live” demo** from frames that simulates real-time streaming using a rolling window of `T` frames

---

## 1) Goal / Core Problem

### Goal
Build a reproducible and debuggable pipeline for **ultrasound frame sequences** that performs **temporal-aware segmentation** for:
- **Needle** (thin, line-like structure, hard to segment per-frame)
- **Anatomy** (larger structures: saphenous nerve, femoral artery, sartorius muscle)

### Core Problems Solved
- CVAT exports masks in **RLE** that must be decoded correctly (foreground/background ordering matters).
- Ultrasound is noisy; temporal context improves stability → maintain a rolling window of frames.
- Needle is small + class imbalance is severe (tiny positive pixels) → thresholding + postprocessing needed.
- Anatomy is heavily imbalanced → optional class weighting supported.

---

## 2) High-Level Architecture

### Data Stages
1. **Raw data** (`data_raw/`)
   - `annotations.xml` (CVAT 1.1 export)
   - `images/` containing frames like: `0_frame_0.jpg`, `8_frame_13.jpg`, etc.
   - In XML, image names look like: `images/<vid>_frame_<fid>.jpg`

2. **Processed data** (`data_processed/`)
   - `masks/needle/*.png` (0/255 binary masks)
   - `masks/anatomy/*.png` (0/85/170/255 encoding: bg/nerve/artery/muscle)
   - `meta/frames.csv` (per-frame metadata)
   - `meta/splits.json` (video splits)
   - `meta/sequences_needle_{train,val,test}.json` (temporal manifests)
   - `meta/sequences_anatomy_{train,val,test}.json` (temporal manifests)
   - `sanity_overlays/*.png` (quick alignment checks)

3. **Runs / artifacts** (`runs/`)
   - `runs/needle_convlstm/best.pt`, `last.pt`
   - `runs/anatomy_convlstm/best.pt`, `run_config.json`
   - `debug_preds/` outputs from inference debug scripts

### Model Stage
- **ConvLSTM-UNet** (`src/models_convlstm_unet.py`)
  - UNet encoder applied per-frame (shared weights)
  - ConvLSTM at bottleneck accumulates temporal memory
  - Decoder uses last-frame skip connections + ConvLSTM hidden state
  - Outputs logits for last frame only

---

## 3) Major Decisions & Rationale

### A) Use temporal model (ConvLSTM-UNet) vs per-frame UNet
**Decision:** ConvLSTM at bottleneck to capture motion/consistency.  
**Why:** Needle visibility fluctuates; temporal memory stabilizes detections.  
**Tradeoff:** More compute and more sensitivity to preprocessing / normalization.

### B) Use CVAT mask RLE decode with correct foreground ordering
**Decision:** Implement proper CVAT RLE decoding (foreground-first / background-first pitfalls).  
**Why:** Early symptoms were “inverted masks” / “mask around needle”. Fix required correct decode assumptions and (for needle/anatomy) correct mapping to global mask.

### C) Needle segmentation treated as binary probability + threshold + postprocess
**Decision:** Needle model outputs sigmoid probabilities → choose threshold `thr` and optionally keep largest component.  
**Why:** Thin structure + false positives appear as secondary blobs.  
**Tradeoff:** Threshold depends on calibration; best `thr` selected via sweep.

### D) Anatomy treated as multiclass argmax (no threshold sweep)
**Decision:** Anatomy uses `CrossEntropyLoss` and `argmax(logits)` inference.  
**Why:** Standard multiclass segmentation approach.  
**Tradeoff:** Severe class imbalance; weights often required.

### E) Normalization is per-sequence (optional)
**Decision:** Dataset loaders normalize input sequences by mean/std when enabled.  
**Why:** Ultrasound brightness varies; simple normalization improves stability.  
**Tradeoff:** If live demo path doesn’t match training normalization, predictions collapse.

---

## 4) Folder Structure

```
(TODO)

````

---

## 5) End-to-End Workflow (Recommended Order)

### Step 0 — Environment
Create/activate env (`ugra`) with Python 3.10 and GPU torch build compatible with your GPU.

Sanity check GPU:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
````

---

### Step 1 — Preprocess CVAT → masks + manifests

**File:** `preprocess.py` (in repo root)

**Produces:**

* `data_processed/masks/needle/*.png`
* `data_processed/masks/anatomy/*.png`
* `data_processed/meta/*.json`
* `data_processed/sanity_overlays/*.png`

**Run:**

```bash
python preprocess.py --xml data_raw\annotations.xml --images_dir data_raw\images --out_dir data_processed --window 8 --stride 1 --val_videos 6 --test_videos 8
```

**Key args:**

* `--xml`: CVAT export
* `--images_dir`: folder containing `<vid>_frame_<fid>.jpg`
* `--out_dir`: default `data_processed`
* `--window`: temporal length `T` used for manifests (8 recommended)
* `--stride`: usually 1
* `--val_videos`, `--test_videos`: split by **video_id** (train is remaining)

**Validation:**
Open a few `data_processed/sanity_overlays/*.png` to confirm overlays align.

---

### Step 2 — Quick sanity checks (optional but recommended)

#### 2.1 Loader sanity (needle)

**File:** `src/check_loader.py`
Ensures dataset returns correct shapes and mask values.

Run:

```bash
python src\check_loader.py
```

Expected prints like:

* `x shape: (B, T, 1, 640, 640)`
* `y shape: (B, 1, 640, 640)`
* `y unique: tensor([0.,1.])`

#### 2.2 Model forward sanity

**File:** `src/check_model.py`
Runs a dummy forward pass and checks output shape.

Run:

```bash
python src\check_model.py
```

Expected:

* `out shape: torch.Size([2, 1, 640, 640])` (needle)
* For anatomy, should be `(B, 4, H, W)` if configured.

---

### Step 3 — Train Needle Temporal Model

**File:** `src/train_needle_temporal.py`

Run:

```bash
python src\train_needle_temporal.py --epochs 30 --batch_size 1 --amp --lr 2e-4 --base 32
```

Outputs:

* `runs/needle_convlstm/best.pt`
* optionally `last.pt`

Notes:

* `--base` controls UNet channel width (32 default).
* `--amp` enables mixed precision on CUDA.

---

### Step 4 — Needle Debug Inference (visual)

**File:** `src/infer_debug.py`

Writes side-by-side images: **GT | Pred** into:

* `runs/needle_convlstm/debug_preds/val/`
* `runs/needle_convlstm/debug_preds/test/`

Run:

```bash
python src\infer_debug.py --split val --thr 0.50 --postprocess largest
python src\infer_debug.py --split test --thr 0.50 --postprocess largest
```

---

### Step 5 — Needle Threshold Sweep (select best thr)

**File:** `src/sweep_thresholds.py`

This computes mean Dice across the split at different thresholds.

Run:

```bash
python src\sweep_thresholds.py --split val --postprocess largest
python src\sweep_thresholds.py --split test --postprocess largest
```

Example results you observed:

* best val around `thr=0.60`
* best test around `thr=0.85`
  (These can shift after annotation fixes.)

---

### Step 6 — Needle “Live” demo from frames (real-time simulation)

**File:** `src/live_needle_demo.py`

**Recommended mode:** Manifest mode (ensures identical normalization / window build as training)

Run (example):

```bash
python src\live_needle_demo.py ^
  --manifest data_processed\meta\sequences_needle_test.json ^
  --window 8 ^
  --ckpt runs\needle_convlstm\best.pt ^
  --thr 0.85 ^
  --postprocess largest ^
  --fps_cap 30
```

**Behavior:**

* iterates sequences from manifest
* builds a rolling window `T`
* runs inference each step
* overlays **GT** and **Pred** on the last frame
* includes pause/resume + toggles

**Important:** Live demo must match training normalization. Using “raw scan mode” can break preds if normalization differs.

---

### Step 7 — Train Anatomy Temporal Model

**File:** `src/train_anatomy_temporal.py`

Run baseline first:

```bash
python src\train_anatomy_temporal.py --amp --epochs 150 --batch_size 1 --lr 2e-4 --T 8 --base 32
```

If imbalance hurts performance, compute weights then train weighted:

1. Estimate weights:

```bash
python src\train_anatomy_temporal.py --calc_class_weights --calc_max_samples 2000
```

2. Train with weights:

```bash
python src\train_anatomy_temporal.py --amp --epochs 150 --class_weights "0.0137,2.3438,1.4731,0.1694"
```

**Notes on `--calc_max_samples`:**

* It’s a *cap* on dataset items scanned to estimate pixel ratios.
* If dataset has <2000 sequences, it scans them all.

---

### Step 8 — Anatomy Debug Inference (visual)

**File:** `src/infer_debug_anatomy.py`

Run:

```bash
python src\infer_debug_anatomy.py --split val
python src\infer_debug_anatomy.py --split test
```

Outputs go to:

* `runs/anatomy_convlstm/debug_preds/{val,test}/`

---

### Step 9 — Combined Live Demo (Needle + Anatomy overlay together)

**File:** `src/live_combined_demo.py`

Expected behavior:

* same frame window drives both models
* overlays both:

  * needle in red
  * anatomy classes in distinct colors
* ensures sync by using same manifest window and last-frame display

Run example:

```bash
python src\live_combined_demo.py ^
  --manifest_needle data_processed\meta\sequences_needle_test.json ^
  --manifest_anatomy data_processed\meta\sequences_anatomy_test.json ^
  --T 8 ^
  --needle_ckpt runs\needle_convlstm\best.pt ^
  --anatomy_ckpt runs\anatomy_convlstm\best.pt ^
  --needle_thr 0.85 ^
  --needle_postprocess largest ^
  --fps_cap 30
```

(If your combined script uses different arg names, align to its CLI.)

---

## 6) File-by-File Explanation (API / Inputs / Outputs)

### `preprocess.py` (root)

**Purpose:** CVAT XML → masks + metadata + temporal manifests
**Inputs:**

* `data_raw/annotations.xml`
* `data_raw/images/*.jpg`
  **Outputs:**
* `data_processed/masks/needle/<stem>.png`
* `data_processed/masks/anatomy/<stem>.png`
* `data_processed/meta/frames.csv`
* `data_processed/meta/splits.json`
* `data_processed/meta/sequences_*_{split}.json`
* `data_processed/sanity_overlays/*.png`
  **Depends on:** consistent filename pattern `<vid>_frame_<fid>.jpg` and XML names `images/<file>`.

---

### `src/models_convlstm_unet.py`

**Purpose:** defines ConvLSTM-UNet model
**Main class:** `ConvLSTMUNet(in_channels=1, base=32, out_channels=1 or 4)`
**I/O:**

* Input: `x` shape `(B, T, 1, H, W)`
* Output:

  * needle: logits `(B, 1, H, W)`
  * anatomy: logits `(B, 4, H, W)`

---

### `src/datasets_temporal.py`

**Purpose:** needle dataset loader from manifest
**Class:** `TemporalNeedleDataset(manifest_path, expected_T=8, normalize=True)`
**Outputs:** `TemporalSample`

* `x`: `(T,1,H,W)` float32
* `y`: `(1,H,W)` float32 {0,1}
* `meta`: dict including `frame_paths`, `mask_path`

**Critical contract:** normalization here must match any “live demo” input path.

---

### `src/datasets_anatomy_temporal.py`

**Purpose:** anatomy dataset loader from manifest
Similar contract to needle but:

* `y` is class map `(H,W)` int64 with classes `{0,1,2,3}`

---

### `src/train_needle_temporal.py`

**Purpose:** trains needle model
**Reads:**

* `data_processed/meta/sequences_needle_train.json`
* `data_processed/meta/sequences_needle_val.json`
  **Writes:**
* `runs/needle_convlstm/best.pt`
* maybe `last.pt`

---

### `src/train_anatomy_temporal.py`

**Purpose:** trains anatomy multiclass model
**Reads:**

* `data_processed/meta/sequences_anatomy_train.json`
* `data_processed/meta/sequences_anatomy_val.json`
  **Writes:**
* `runs/anatomy_convlstm/best.pt`
* `run_config.json` (if enabled in your version)

Supports:

* `--calc_class_weights` estimation
* `--class_weights "w0,w1,w2,w3"` for CrossEntropyLoss

---

### `src/postprocess.py`

**Purpose:** postprocessing utilities
**Common:** keep one component to reduce false positives:

* `keep_one_component(mask, mode="largest"|"longest", min_area=...)`

Used by:

* `infer_debug.py`
* `sweep_thresholds.py`
* live demos (optional)

---

### `src/infer_debug.py` (needle)

**Purpose:** write GT|Pred side-by-side PNGs for quick inspection
**Inputs:** ckpt + manifest
**Outputs:** `runs/needle_convlstm/debug_preds/<split>/*.png`

---

### `src/infer_debug_anatomy.py`

**Purpose:** same idea for anatomy outputs

---

### `src/sweep_thresholds.py`

**Purpose:** choose best needle threshold by evaluating mean Dice across a split
**Inputs:** ckpt + manifest split
**Outputs:** printed metrics (`mean_dice`) per threshold

---

### `src/live_needle_demo.py`

**Purpose:** real-time simulation from frame sequences (OpenCV UI)
**Modes:**

* Manifest mode (recommended): `--manifest ...`
* Raw mode: `--video_id ...` (more fragile if preprocessing differs)

**Key contracts:**

* if training normalized inputs, live demo must normalize similarly
* if using manifests, window is exact same T frames as training examples

---

### `src/live_combined_demo.py`

**Purpose:** combined overlay of needle + anatomy in one UI
**Contracts:** both models must be fed the same temporal window and same last-frame display.

---

### `src/check_loader.py`

**Purpose:** confirm dataset output shapes and mask values

---

### `src/check_model.py`

**Purpose:** confirm model forward pass and output shape

---

### `src/debug_parity_one_sample.py`

**Purpose:** “parity” debugging between two codepaths (typically:

* confirm live demo inference matches infer_debug inference on the exact same sample)

Use when predictions diverge.

---

## 7) Common Debug Insights / Failure Modes

### A) “Pred pixels are always 0”

Almost always one of:

* **normalization mismatch** (training vs live path)
* wrong ckpt loaded / wrong `base` or `out_channels`
* using raw frames with different preprocessing than manifest pipeline
* threshold too high (but your prob stats showed max ~0.12 in failure mode → indicates model is outputting low everywhere)

Fix approach:

* run `debug_parity_one_sample.py` to compare infer_debug vs live on exact same sample
* ensure live demo uses manifest mode and applies same normalization

### B) “Masks inverted / needle is black line inside white block”

That indicates CVAT RLE decode mismatch (foreground/background ordering).
This was fixed in preprocessing so both needle and anatomy masks are correct now.

### C) Anatomy model “sucks”

Typical causes:

* severe imbalance (background dominates; nerve/artery tiny)
* insufficient data or inconsistent annotations
* under-capacity (`base` too small)
* class weights too extreme or missing

Next actions:

* train baseline then weighted
* check per-class dice (not just mean)
* consider cropping / ROI training later (optional)

---

## 8) Parameters Glossary (what to touch vs keep default)

### Model capacity

* `--base`: channel width multiplier for UNet (32 default). Higher = more capacity + VRAM.
* `--T` / `--window`: temporal length (8 default). Higher = more context + compute.

### Training control

* `--epochs`: longer training may help, but watch overfit.
* `--lr`: 2e-4 baseline.
* `--batch_size`: 1 is OK for 640² + temporal; increase only if VRAM allows.
* `--amp`: recommended on GPU for speed/VRAM.

### Loader performance

* `--num_workers`: dataloader workers. Start with `0` on Windows, try `2-4` if stable.

### Needle inference tuning

* `--thr`: threshold for converting probabilities to binary
* `--postprocess`: `largest` to keep the single most plausible needle component

### Live demo speed

* `--fps_cap`: maximum FPS (0 or omit to run as fast as possible). In practice, UI loop will still limit.

---

## 9) Recommended “Production” Command Sets

### Full rebuild + needle training + selection + live demo

```bash
python preprocess.py --xml data_raw\annotations.xml --images_dir data_raw\images --out_dir data_processed --window 8 --stride 1 --val_videos 6 --test_videos 8

python src\train_needle_temporal.py --epochs 30 --batch_size 1 --amp --lr 2e-4 --base 32

python src\sweep_thresholds.py --split val --postprocess largest
python src\sweep_thresholds.py --split test --postprocess largest

python src\live_needle_demo.py --manifest data_processed\meta\sequences_needle_test.json --window 8 --ckpt runs\needle_convlstm\best.pt --thr 0.85 --postprocess largest --fps_cap 30
```

### Anatomy training + debug

```bash
python src\train_anatomy_temporal.py --amp --epochs 150 --batch_size 1 --lr 2e-4 --T 8 --base 32

python src\infer_debug_anatomy.py --split val
python src\infer_debug_anatomy.py --split test
```

### Combined live demo (after anatomy is decent)

```bash
python src\live_combined_demo.py --T 8 --needle_ckpt runs\needle_convlstm\best.pt --anatomy_ckpt runs\anatomy_convlstm\best.pt --needle_thr 0.85 --needle_postprocess largest --fps_cap 30
```

---

## 10) Current Status / Next Development Steps

### What’s working

* Preprocess pipeline producing correct masks (needle lines, anatomy regions)
* Needle temporal model training + threshold sweep + live demo (GT + Pred visible)
* Debug tooling (infer_debug + sweep_thresholds)

### Known gaps / next steps

1. **Improve anatomy model quality**

   * run baseline + weighted training
   * evaluate per-class dice and failure cases
   * consider mild augmentations later (optional)

2. **Combined UI**

   * ensure combined demo uses the same manifest window for both models
   * allow toggles for needle/anatomy overlays

3. **Performance / optimization**

   * optional: run needle less frequently than display FPS (e.g., infer every N frames)
   * optional: inference batching / torch.compile (later)

4. **Annotation policy**

   * needle annotations should correspond to when the needle is actually visible
   * reduce ambiguous faint needle labeling (helps model calibration)

---

## 11) Handoff Notes (for another AI)

* Always prefer **manifest mode** for inference parity with training.
* If predictions “die” in live demo, first suspect **normalization mismatch**.
* `preprocess.py` is the single source of truth for:

  * frame parsing rules
  * mask encoding
  * split assignment
  * temporal manifest creation

---

```
::contentReference[oaicite:0]{index=0}
```
