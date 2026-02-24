# UGRA — Temporal Ultrasound Segmentation (Needle + Anatomy)

UGRA trains and runs two temporal segmentation models on ultrasound frame sequences:

- Needle segmentation (binary): ConvLSTM-UNet
- Anatomy segmentation (multi-class: bg / nerve / artery / muscle): 2D Swin encoder + ConvLSTM + decoder

Both tasks are causal last-frame prediction: each sample is a window of T frames, and the target is the mask for the final frame.

## 1) Current Project Status

- Needle pipeline is operational end-to-end (preprocess -> train -> debug/live inference).
- Anatomy pipeline is operational with Swin + ConvLSTM training and debug inference.
- Combined live demo currently runs on looped file frames (simulated live stream), which is the intended behavior for now.

## 2) Repository Structure (Key Files)

- preprocess.py
  - CVAT XML -> masks + metadata + temporal manifests.
- src/datasets_temporal.py
  - Needle temporal dataset.
- src/datasets_anatomy_temporal.py
  - Anatomy temporal dataset.
- src/models_convlstm_unet.py
  - ConvLSTM primitives + UNet blocks (shared components, needle model).
- src/models_anatomy_swin_convlstm.py
  - Anatomy model (2D Swin per frame, ConvLSTM temporal bottleneck, UNet-like decoder).
- src/train_needle_temporal.py
  - Needle training (DiceCE + validation Dice metric).
- src/train_anatomy_temporal.py
  - Anatomy training (Swin+ConvLSTM or ConvLSTM-only; CE or MONAI DiceCE + per-class Dice reporting).
- src/infer_debug.py
  - Needle overlay export for sanity checks.
- src/infer_debug_needle.py
  - Alias entrypoint for needle debug overlays (same behavior as infer_debug.py).
- src/infer_debug_anatomy.py
  - Anatomy overlay export for sanity checks.
- src/live_needle_demo.py
  - Needle live-style demo.
- src/live_combined_demo.py
  - Combined needle + anatomy live-style overlay demo.
- src/compare_runs.py
  - Compare multiple training runs side-by-side (ranked table of Dice, FP, loss, settings).

## 3) Data and Labels

Input sources:

- data_raw/annotations.xml
- data_raw/images/{video_id}_frame_{frame_id}.jpg

Generated outputs:

- data_processed/masks/needle/*.png (0 or 255; loaded as 0/1)
- data_processed/masks/anatomy/*.png (0, 85, 170, 255 -> class IDs 0..3)
- data_processed/meta/frames.csv
- data_processed/meta/splits.json
- data_processed/meta/sequences_needle_{train|val|test}.json
- data_processed/meta/sequences_anatomy_{train|val|test}.json

## 4) Environment

Conda environment file is provided at environment.yml.

Windows (PowerShell):

```powershell
conda env create -f environment.yml
conda activate ugra
python -c "import torch, monai; print('torch', torch.__version__, 'monai', monai.__version__)"
```

## 5) End-to-End Workflow

### Step A: Preprocess

Example:

```powershell
python preprocess.py `
  --xml data_raw\annotations.xml `
  --images_dir data_raw\images `
  --out_dir data_processed `
  --needle_window 8 `
  --anatomy_window 4 `
  --needle_min_frames_for_temporal 8 `
  --anatomy_min_frames_for_temporal 4 `
  --stride 1 `
  --val_videos 6 `
  --test_videos 8
```

Quick manifest sanity:

```powershell
python -c "import json;print('needle_train',len(json.load(open(r'data_processed\meta\sequences_needle_train.json'))))"
python -c "import json;print('anatomy_train',len(json.load(open(r'data_processed\meta\sequences_anatomy_train.json'))))"
python -c "import json;print('anatomy_val',len(json.load(open(r'data_processed\meta\sequences_anatomy_val.json'))))"
```

### Step B: Train Needle

```powershell
python src\train_needle_temporal.py --amp --epochs 50 --T 8 --batch_size 1 --lr 2e-4 --base 32
```

### Step C: Train Anatomy (Swin + ConvLSTM)

```powershell
python src\train_anatomy_temporal.py --amp --epochs 50 --T 4 --batch_size 1 --lr 2e-4 --swin_embed_dim 48 --swin_window_size 8 --use_checkpoint
```

Recommended for speckle reduction (MONAI DiceCE):

```powershell
python src\train_anatomy_temporal.py --amp --epochs 50 --T 4 --batch_size 1 --lr 2e-4 --swin_embed_dim 48 --swin_window_size 8 --use_checkpoint --loss dicece --lambda_dice 1.0 --lambda_ce 1.0
```

Recommended for strong class imbalance (MONAI Generalized Dice + CE):

```powershell
python src\train_anatomy_temporal.py --amp --epochs 50 --T 4 --batch_size 1 --lr 2e-4 --swin_embed_dim 48 --swin_window_size 8 --use_checkpoint --loss gdicece --lambda_dice 0.8 --lambda_ce 0.6
```

ConvLSTM-only anatomy ablation (no Swin encoder):

```powershell
python src\train_anatomy_temporal.py --amp --epochs 50 --T 4 --batch_size 1 --lr 2e-4 --arch convlstm --convlstm_base 32 --loss dicece --lambda_dice 1.0 --lambda_ce 1.0
```

Optional class-weight recommendation for anatomy:

```powershell
python src\train_anatomy_temporal.py --T 4 --calc_class_weights --calc_max_samples 2000
```

### Step D: Debug Predictions (offline overlays)

Debug scripts auto-select the newest run checkpoint (`run_*/best.pt`) and auto-resolve T from checkpoint metadata.
Each exported PNG is a 3-panel triptych: **Raw | GT overlay | Prediction overlay**.

Needle:

```powershell
python src\infer_debug_needle.py --split val --n 12
```

Anatomy:

```powershell
python src\infer_debug_anatomy.py --split val --n 12 --swin_embed_dim 48 --swin_window_size 8 --use_checkpoint
```

Point at a specific run folder's checkpoint:

```powershell
python src\infer_debug_anatomy.py --ckpt runs\anatomy_swin_convlstm\run_YYYYMMDD_HHMMSS_swin_convlstm\best.pt --split val --n 12
```

Note: anatomy postprocess defaults to none. Use postprocess flags only when you explicitly want filtering.

If masks are still cloudy/speckled, add confidence gating (example):

```powershell
python src\infer_debug_anatomy.py --split val --n 12 --swin_embed_dim 48 --swin_window_size 8 --use_checkpoint --postprocess largest_per_class --min_area_nerve 40 --min_area_artery 40 --min_area_muscle 120 --min_conf 0.55
```

### Step D.5: Compare Runs

After training several experiments, compare them side-by-side:

```powershell
python src\compare_runs.py --root runs\anatomy_swin_convlstm
python src\compare_runs.py --root runs\anatomy_swin_convlstm --sort fp_rate_nerve
python src\compare_runs.py --root runs\needle_convlstm
```

This prints a ranked table with best Dice, per-class Dice, FP rates, loss, class weights, and architecture for each run.

### Step E: Combined Live-Style Demo

Note: this is looped-frame simulation from data_raw/images for the selected video_id.
By default GT overlay is OFF in live combined mode; press g to toggle GT on/off.
Useful keys: q quit, space pause, p needle on/off, a anatomy on/off, [ and ] adjust needle threshold.

Zero-config default run:

```powershell
python src\live_combined_demo.py
```

Defaults use:
- video_id=8
- needle_ckpt=auto (newest runs\needle_convlstm\run_*\best.pt, fallback runs\needle_convlstm\best.pt)
- anat_ckpt=auto (newest runs\anatomy_swin_convlstm\run_*\best.pt, fallback runs\anatomy_swin_convlstm\best.pt)
- needle_window=auto from needle checkpoint T (`--needle_window 0`)
- anat_window=auto from anatomy checkpoint T (`--anat_window 0`)
- anatomy postprocess is OFF by default

```powershell
python src\live_combined_demo.py `
  --video_id 8 `
  --needle_window 8 `
  --anat_window 4 `
  --needle_ckpt runs\needle_convlstm\run_YYYYMMDD_HHMMSS_needle\best.pt `
  --anat_ckpt runs\anatomy_swin_convlstm\run_YYYYMMDD_HHMMSS_swin_convlstm\best.pt `
  --needle_base 32 `
  --needle_thr 0.85 `
  --anat_swin_embed_dim 48 `
  --anat_swin_window_size 8 `
  --anat_use_checkpoint
```

Optional anatomy filtering (EXTRA; disabled by default):

```powershell
python src\live_combined_demo.py `
  --anat_postprocess largest_per_class `
  --anat_min_area_nerve 40 `
  --anat_min_area_artery 40 `
  --anat_min_area_muscle 120 `
  --anat_min_conf 0.55
```

## 6) Parameter Guide (Especially for RTX 5070 Ti 16GB)

Primary VRAM knobs:

- Anatomy T (temporal window): biggest temporal memory lever
- swin_embed_dim: biggest model-width lever
- use_checkpoint: lowers VRAM at speed cost
- batch_size: keep at 1 for anatomy unless clearly stable

Recommended anatomy training ladder:

1. Stable baseline (start here)
   - T=4, batch_size=1, swin_embed_dim=32, swin_window_size=8, use_checkpoint=true, amp=true
2. Quality step-up
   - increase swin_embed_dim to 48 (keep T=4)
3. Temporal step-up (if VRAM allows)
   - increase T from 4 -> 6
4. Only then consider larger img_size or higher embed_dim

Needle usually fits comfortably with:

- T=8, base=32, batch_size=1, amp=true

## 7) Known Pitfalls and Fixes

1) Empty anatomy validation split

- Symptom: training aborts with empty val dataset.
- Cause: split/min_frames/window constraints remove all val windows.
- Fix: adjust val_videos, anatomy_window (T), and anatomy_min_frames_for_temporal.

2) Swin dimensional constraints

- For this project: 2D Swin is used.
- Keep img_size divisible by patch_size and usually by window_size to avoid downstream shape/window issues.

3) Mixed precision + lazy module construction

- Anatomy model builds ConvLSTM/decoder lazily on first forward.
- Those modules must be moved to the active device during lazy build.
- ConvLSTM execution is forced to float32 with autocast disabled locally for robustness.

## 8) CLI Cheatsheet

Needle train:

- --T: temporal window length
- --base: UNet base channels
- --val_thr: threshold used for reported val Dice

Anatomy train:

- --T: temporal window length
- --arch: swin_convlstm (default) or convlstm
- --convlstm_base: base channels for convlstm architecture
- --swin_embed_dim: model width
- --swin_window_size: spatial window (internally converted to 2D tuple)
- --patch_size: Swin patch size
- --use_checkpoint: lower VRAM, slower
- --loss: ce, dicece, gdice, or gdicece
- --lambda_dice and --lambda_ce: DiceCE weighting controls
- --class_weights: optional CE weights in order bg,nerve,artery,muscle

Combined demo:

- --needle_window and --anat_window must match each model's trained T
- --needle_base and anatomy Swin args should match training config
- --anat_postprocess: none, largest_per_class, longest_per_class
- --anat_min_area_nerve / --anat_min_area_artery / --anat_min_area_muscle: speckle filtering per class
- --anat_min_conf: confidence gating before postprocess (higher removes weak false positives)
- The script validates CLI args against checkpoint metadata (and nearby run_config.json when present) and raises a clear error on mismatch.

## 9) Training Artifacts (auto-generated)

Each training run creates a timestamped subfolder under the task output root.
All files live **inside the run subfolder only** — no files are duplicated to the root.

Needle root: `runs/needle_convlstm/run_YYYYMMDD_HHMMSS_needle/`
Anatomy root: `runs/anatomy_swin_convlstm/run_YYYYMMDD_HHMMSS_swin_convlstm/`

Per-run folder includes:
- `best.pt` and `last.pt` — model checkpoints
- `run_config.json` — full training config (args + model_cfg)
- `command.txt` — exact command used to start the run
- `metrics.csv` — epoch-wise metrics (loss, Dice, FP rates, LR)
- `summary.json` — best epoch and best validation metric
- `training_curves.jpg` — loss + Dice + FP plots
- `confusion_matrix.csv` and `confusion_matrix.jpg` — final validation confusion matrix (anatomy only)

Root folder keeps only:
- `latest_run.txt` — path to the most recent run subfolder (used by auto-checkpoint resolution)

To use a specific run's checkpoint, pass the full path:
```powershell
--ckpt runs\anatomy_swin_convlstm\run_20260216_124756_swin_convlstm\best.pt
```
Or use `--ckpt auto` (default) to auto-select the newest `run_*/best.pt`.

## 10) Understanding Loss & Class Imbalance Options (Intuitive Guide)

The anatomy task has extreme class imbalance: background dominates (~88% of pixels), muscle is moderate (~10%), while nerve (~0.6%) and artery (~1.3%) are tiny. The loss function determines how the model "penalizes" mistakes during training — different losses handle this imbalance differently.

### Loss types (`--loss`)

| Loss | What it does intuitively |
|------|--------------------------|
| `ce` | **Cross-Entropy**: treats every pixel equally. Background correctly predicted = big reward. Tiny classes get drowned out — the model can score well by predicting "all background". Fast to train but often under-segments small structures. |
| `dicece` | **Dice + CE combined**: Dice measures overlap *per class* (so nerve Dice is computed only among nerve pixels, not diluted by background). CE keeps predictions sharp. Together they balance region-level awareness with pixel-level precision. Best general starting point. |
| `gdice` | **Generalized Dice**: like Dice but automatically down-weights large classes (background) and up-weights tiny ones (nerve, artery) based on their pixel counts *in each batch*. More aggressive than standard Dice for imbalance. Can be unstable alone. |
| `gdicece` | **Generalized Dice + CE**: combines the imbalance-aware Dice with the stable CE. Usually the best choice when nerve/artery are very small. `--lambda_dice` and `--lambda_ce` control the balance: higher lambda_dice = more focus on class overlap, higher lambda_ce = more focus on per-pixel correctness. |

### Class weights (`--class_weights`)

Separate from the loss function. This tells the CE term "pay X times more attention to mistakes on this class."

```powershell
# First, estimate optimal weights from your data:
python src\train_anatomy_temporal.py --calc_class_weights

# Output example: suggested --class_weights: 0.0175,2.6341,1.1871,0.1613
#                                             bg     nerve  artery muscle
# Meaning: nerve mistakes are penalized 150x more than background mistakes
```

You can combine class weights with any loss that uses CE internally (`ce`, `dicece`, `gdicece`).

### Lambda controls (`--lambda_dice`, `--lambda_ce`)

These scale the two loss terms when using `dicece` or `gdicece`:
- `--lambda_dice 1.0 --lambda_ce 1.0` = equal weight (default)
- `--lambda_dice 1.0 --lambda_ce 0.5` = Dice dominates; more region-focused, fewer speckles but potentially weaker boundaries
- `--lambda_dice 0.5 --lambda_ce 1.0` = CE dominates; sharper boundaries but potentially more false positives on small classes

### LR scheduling (`--scheduler`)

| Option | How LR changes | When to use |
|--------|---------------|-------------|
| `none` | Constant LR throughout training (default) | All runs — **safe default, does not change existing behaviour** |
| `cosine` | Linearly warms up for `--warmup_epochs` (default 5) then smoothly decays to near-zero via cosine curve | Optional for longer runs (≥80 epochs) — may help squeeze extra quality but not required |

**Safety note:** `--scheduler none` is the default and behaves identically to all prior runs. Adding `--scheduler cosine` is purely opt-in — it will never activate unless you explicitly pass it.

### Estimating class weights

```powershell
python src\train_anatomy_temporal.py --calc_class_weights
```

Scans up to 2,000 training masks, counts pixels per class, and computes inverse-frequency weights:
- Output example: `0.0175,2.6341,1.1871,0.1613` (bg, nerve, artery, muscle)
- Nerve errors are penalized ~150× more than background errors

### Recommended experiment sequence

```powershell
# 1) Baseline — plain CE (fast, likely under-segments nerve/artery)
python src\train_anatomy_temporal.py --amp --epochs 50 --loss ce

# 2) DiceCE — usually a big improvement over CE
python src\train_anatomy_temporal.py --amp --epochs 50 --loss dicece

# 3) GDiceCE — specifically helps tiny classes (nerve, artery)
python src\train_anatomy_temporal.py --amp --epochs 50 --loss gdicece --lambda_dice 0.8 --lambda_ce 0.6

# 4) GDiceCE + class weights — maximum imbalance compensation
python src\train_anatomy_temporal.py --amp --epochs 50 --loss gdicece --lambda_dice 0.8 --lambda_ce 0.6 --class_weights 0.0175,2.6341,1.1871,0.1613

# 5) Compare all runs
python src\compare_runs.py --root runs\anatomy_swin_convlstm
python src\compare_runs.py --root runs\anatomy_swin_convlstm --sort dice_nerve

# 6) Visual check on best run
python src\infer_debug_anatomy.py --split val --n 5 --out_dir runs\anatomy_swin_convlstm\debug_preds\val

# 7) Live demo (both models)
python src\live_combined_demo.py
```

### How to read the comparison table

| Column | What it means | Good value |
|--------|--------------|------------|
| `dice` | Average Dice across nerve + artery + muscle (excludes background). **Primary quality metric.** | Higher is better (0.0–1.0) |
| `iou` | Mean Intersection-over-Union (excludes background). Stricter than Dice — penalizes both FP and FN. | Higher is better |
| `d_nrv/d_art/d_mus` | Per-class Dice (nerve, artery, muscle). | Higher is better |
| `p_nrv` | Nerve precision: of all pixels the model called "nerve", how many truly were nerve. | Higher = fewer false alarms |
| `r_nrv` | Nerve recall: of all true nerve pixels, how many did the model find. | Higher = fewer missed nerve |
| `fp_nrv/fp_art` | False positive rate per class. | Lower is better (speckle indicator) |
| `loss_f` | Training loss at last epoch. Very high = didn't converge. Very low = possible overfitting. | Check convergence |
| `ep` | Epoch of best checkpoint. If very early, model may be underfitting later. | Should be >50% of total epochs |
| `sched` | LR scheduler used. | `none` is safe default |

### How to read training_curves.jpg

The plot has 9 panels (3×3 grid):

**Row 1 — Core metrics:**
1. **Train Loss** — should decrease steadily. Flat early = model not learning (try different loss or LR).
2. **Validation Dice** — should increase. Peak then drop = overfitting.
3. **Validation IoU** — stricter version of Dice. Same trend expected but lower absolute values.

**Row 2 — Reliability metrics:**
4. **FP Rate** — should stay low. Rising FP + rising Dice = model hallucinating predictions.
5. **Precision (per class)** — percentage of predictions that are correct. Low precision = too many false positives.
6. **Recall (per class)** — percentage of GT that was found. Low recall = model missing structures.

**Row 3 — Diagnostic panels:**
7. **Nerve Deep-Dive** — Dice, FP, precision, recall for nerve (hardest class) on one plot.
8. **Learning Rate** — shows the LR schedule (flat line when using `none`; smooth decay with `cosine`).
9. **Dice vs IoU** — correlation check; if they diverge significantly, investigate class-specific issues.

### How to read confusion_matrix.jpg

- Rows = ground truth class; Columns = predicted class.
- Diagonal values should be close to 1.0 (correct predictions).
- Off-diagonal values show systematic confusions (e.g., nerve row / bg column = model misses nerve as background = under-segmentation).

## 11) Copilot Context Appendix (for future coding sessions)

Project invariants to preserve:

- Last-frame causal prediction for both tasks.
- Temporal window manifests are authoritative sample definitions.
- Needle mask is binary; anatomy mask is 4-class categorical.
- Live demo path uses simulated stream from frame files.
- All training artifacts live inside run subfolders only (no root-level copies).
- Debug overlays are 3-panel: Raw | GT | Prediction.
- T defaults to 0 (auto-resolve from checkpoint or manifest) in all scripts.
- LR scheduler defaults to none (constant); cosine available as opt-in for longer runs.

When editing architecture/training code, check these files together:

- src/models_convlstm_unet.py
- src/models_anatomy_swin_convlstm.py
- src/datasets_anatomy_temporal.py
- src/train_needle_temporal.py
- src/train_anatomy_temporal.py
- src/live_combined_demo.py
- src/compare_runs.py

High-value maintenance tasks next:

- Add optional gradient accumulation to anatomy training.
- Add manifest stats utility (per-split class balance and sequence counts by video).
