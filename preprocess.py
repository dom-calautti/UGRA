# preprocess.py
# Phase 1 preprocessing for CVAT 1.1 (images) -> masks + metadata + temporal sequence manifests
#
# Outputs:
#   data_processed/
#     masks/anatomy/*.png   (0=bg, 1=nerve, 2=artery, 3=muscle)
#     masks/needle/*.png    (0=bg, 1=needle)
#     meta/frames.csv
#     meta/splits.json
#     meta/sequences_*.json
#     sanity_overlays/*.png

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

# filename format: "<video>_frame_<frame>.jpg"
FILENAME_RE = re.compile(r"^(?P<vid>\d+)_frame_(?P<fid>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_vid_fid(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return None, None
    return int(m.group("vid")), int(m.group("fid"))


def decode_cvat_rle(rle_str: str, w: int, h: int, foreground_first: bool) -> np.ndarray:
    """
    Decode CVAT 1.1 RLE into binary mask (h,w).

    Some labels export as foreground-first, others background-first.
    We control this via foreground_first.

    Returns: uint8 mask with values {0,1}
    """
    runs = [int(x) for x in rle_str.split(",") if x.strip()]
    total = w * h
    flat = np.zeros(total, dtype=np.uint8)

    idx = 0
    val = 1 if foreground_first else 0
    for run in runs:
        if idx >= total:
            break
        end = min(idx + run, total)
        if val == 1:
            flat[idx:end] = 1
        idx = end
        val = 1 - val

    return flat.reshape((h, w))


def paste_local_mask(global_mask: np.ndarray, local_mask: np.ndarray, left: int, top: int, value: int):
    """
    Paste local binary mask into global mask, setting pixels==1 to `value`.
    """
    H, W = global_mask.shape[:2]
    mh, mw = local_mask.shape[:2]

    x1, y1 = left, top
    x2, y2 = left + mw, top + mh

    # clamp to global bounds
    x1c, x2c = max(0, x1), min(W, x2)
    y1c, y2c = max(0, y1), min(H, y2)
    if x2c <= x1c or y2c <= y1c:
        return

    # corresponding crop in local
    lx1 = x1c - x1
    ly1 = y1c - y1
    lx2 = lx1 + (x2c - x1c)
    ly2 = ly1 + (y2c - y1c)

    local_crop = local_mask[ly1:ly2, lx1:lx2]
    region = global_mask[y1c:y2c, x1c:x2c]
    region[local_crop == 1] = value
    global_mask[y1c:y2c, x1c:x2c] = region


def save_mask(mask: np.ndarray, out_path: Path, mode: str):
    """
    mode:
      - "needle": expects values 0/1 -> saves 0/255
      - "anatomy": expects values 0..3 -> saves 0/85/170/255
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "needle":
        vis = (mask.astype(np.uint8) * 255)
    elif mode == "anatomy":
        # 0->0, 1->85, 2->170, 3->255
        vis = (mask.astype(np.uint8) * 85)
    else:
        raise ValueError("mode must be 'needle' or 'anatomy'")

    Image.fromarray(vis, mode="L").save(out_path)


def make_overlay(image_path: Path, mask: np.ndarray, out_path: Path, mode: str):
    """
    Sanity overlay to verify masks align with pixels.
    """
    img = Image.open(image_path).convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.2)
    arr = np.array(img).astype(np.uint8)

    if mode == "needle":
        needle = mask > 0
        arr[needle, 0] = 255
        arr[needle, 1] = (arr[needle, 1] * 0.4).astype(np.uint8)
        arr[needle, 2] = (arr[needle, 2] * 0.4).astype(np.uint8)
    else:
        nerve = mask == 1
        artery = mask == 2
        muscle = mask == 3
        arr[nerve, 0] = 255
        arr[artery, 1] = 255
        arr[muscle, 2] = 255

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def build_sequences(rows, split_name: str, window: int, stride: int, target: str, needle_pos_only: bool, min_frames: int):
    """
    Build causal temporal windows ending at frame i (predict last frame).
    target: "anatomy" or "needle"
    needle_pos_only: if True, only sequences whose target frame has needle.
    min_frames: videos with < min_frames are skipped.
    """
    sequences = []

    # group by video
    vids = {}
    for r in rows:
        vids.setdefault(r["video_id"], []).append(r)

    for vid, vrows in vids.items():
        vrows.sort(key=lambda x: x["frame_id"])
        if len(vrows) < min_frames:
            continue

        n = len(vrows)
        for i in range(window - 1, n, stride):
            tgt = vrows[i]
            if target == "needle" and needle_pos_only and tgt["has_needle"] != 1:
                continue

            window_rows = vrows[i - window + 1: i + 1]

            sequences.append({
                "video_id": vid,
                "split": split_name,
                "target_frame_id": tgt["frame_id"],
                "frames": [w["image_path"] for w in window_rows],
                "target_mask": tgt["needle_mask_path"] if target == "needle" else tgt["anatomy_mask_path"],
                "has_needle": tgt["has_needle"],
            })

    return sequences


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[DONE] Wrote {path} ({len(obj)} sequences)")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--xml", default=r"data_raw\annotations.xml", help="Path to CVAT 1.1 annotations.xml")
    ap.add_argument("--images_dir", default=r"data_raw\images", help="Directory containing frame images")
    ap.add_argument("--out_dir", default=r"data_processed", help="Output directory (data_processed)")

    ap.add_argument("--val_videos", default="", help="Comma-separated video ids for validation (optional)")
    ap.add_argument("--test_videos", default="", help="Comma-separated video ids for test (optional)")

    ap.add_argument("--needle_label", default="Needle", help="Exact CVAT label name for needle masks")
    ap.add_argument("--nerve_label", default="Saphenous Nerve", help="Exact CVAT label name for nerve")
    ap.add_argument("--artery_label", default="Femoral Artery", help="Exact CVAT label name for artery")
    ap.add_argument("--muscle_label", default="Sartorius Muscle", help="Exact CVAT label name for muscle")

    # Default window + per-task overrides
    ap.add_argument("--window", type=int, default=8, help="Default temporal window length T")
    ap.add_argument("--needle_window", type=int, default=0, help="Override T for needle sequences (0 = use --window)")
    ap.add_argument("--anatomy_window", type=int, default=0, help="Override T for anatomy sequences (0 = use --window)")

    ap.add_argument("--stride", type=int, default=1, help="Stride for sequence generation")
    ap.add_argument("--min_frames_for_temporal", type=int, default=8, help="Default: skip videos shorter than this for sequences")
    ap.add_argument("--needle_min_frames_for_temporal", type=int, default=0, help="Override min frames for needle (0 = use default)")
    ap.add_argument("--anatomy_min_frames_for_temporal", type=int, default=0, help="Override min frames for anatomy (0 = use default)")

    ap.add_argument("--overlay_samples", type=int, default=15, help="How many overlays to save per type")

    args = ap.parse_args()

    xml_path = Path(args.xml)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path.resolve()}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

    meta_dir = out_dir / "meta"
    masks_anatomy_dir = out_dir / "masks" / "anatomy"
    masks_needle_dir = out_dir / "masks" / "needle"
    overlay_dir = out_dir / "sanity_overlays"
    meta_dir.mkdir(parents=True, exist_ok=True)

    print("[DEBUG] xml =", xml_path.resolve())
    print("[DEBUG] images_dir =", images_dir.resolve())
    print("[DEBUG] example jpgs =", [p.name for p in list(images_dir.glob("*.jpg"))[:5]])

    val_videos = {int(x) for x in args.val_videos.split(",") if x.strip()}
    test_videos = {int(x) for x in args.test_videos.split(",") if x.strip()}

    anatomy_map = {
        args.nerve_label: 1,
        args.artery_label: 2,
        args.muscle_label: 3,
    }

    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print("[DEBUG] root tag:", root.tag)
    print("[DEBUG] image nodes found:", len(root.findall(".//image")))
    labels_found = sorted({m.attrib.get("label", "") for m in root.findall(".//mask")})
    print("[INFO] Labels found:", labels_found)

    rows = []
    missing_imgs = 0
    skipped = []

    # Process each image node
    for img_node in root.findall(".//image"):
        xml_name = img_node.attrib.get("name", "")
        basename = Path(xml_name).name  # strips "images/" prefix from XML

        vid, fid = parse_vid_fid(basename)
        if vid is None:
            skipped.append(xml_name)
            if len(skipped) <= 25:
                print(f"[WARN] Skipping filename that doesn't match pattern: {xml_name}")
            continue

        img_path = images_dir / basename
        if not img_path.exists():
            missing_imgs += 1
            if missing_imgs <= 10:
                print(f"[DEBUG] Missing image on disk: {img_path}")
            continue

        W = int(img_node.attrib["width"])
        H = int(img_node.attrib["height"])

        anatomy_mask = np.zeros((H, W), dtype=np.uint8)
        needle_mask = np.zeros((H, W), dtype=np.uint8)

        for m in img_node.findall("./mask"):
            label = m.attrib.get("label", "")
            rle = (m.attrib.get("rle", "") or "").strip()
            if not rle:
                continue

            left = int(float(m.attrib.get("left", "0")))
            top = int(float(m.attrib.get("top", "0")))
            mw = int(float(m.attrib.get("width", "0")))
            mh = int(float(m.attrib.get("height", "0")))
            if mw <= 0 or mh <= 0:
                continue

            is_needle = (label == args.needle_label)
            local = decode_cvat_rle(rle, mw, mh, foreground_first=is_needle)

            if label in anatomy_map:
                paste_local_mask(anatomy_mask, local, left, top, anatomy_map[label])

            if is_needle:
                # needle flip (this matched your "needle looks right now" behavior)
                local = 1 - local
                paste_local_mask(needle_mask, local, left, top, 1)

        has_needle = int(needle_mask.sum() > 0)

        # split by video
        if vid in test_videos:
            split = "test"
        elif vid in val_videos:
            split = "val"
        else:
            split = "train"

        stem = Path(basename).stem
        anat_path = masks_anatomy_dir / f"{stem}.png"
        need_path = masks_needle_dir / f"{stem}.png"

        save_mask(anatomy_mask, anat_path, mode="anatomy")
        save_mask(needle_mask, need_path, mode="needle")

        rows.append({
            "video_id": vid,
            "frame_id": fid,
            "filename": basename,
            "xml_name": xml_name,
            "image_path": str(img_path.resolve()),
            "anatomy_mask_path": str(anat_path.resolve()),
            "needle_mask_path": str(need_path.resolve()),
            "has_needle": has_needle,
            "split": split,
            "width": W,
            "height": H,
        })

    (meta_dir / "skipped_files.txt").write_text("\n".join(sorted(set(skipped))))
    print(f"[INFO] Skipped {len(skipped)} non-video-style filenames. List saved to meta/skipped_files.txt")

    if missing_imgs:
        print(f"[WARN] {missing_imgs} images referenced in XML were missing on disk.")

    if not rows:
        raise RuntimeError("No frames parsed into rows. Check images_dir, XML image name patterns, and missing image debug output.")

    rows.sort(key=lambda r: (r["video_id"], r["frame_id"]))

    # Write frames.csv
    frames_csv = meta_dir / "frames.csv"
    with open(frames_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[DONE] Wrote", frames_csv)

    # Resolve windows/min-frames (defaults + per-task overrides)
    stride = args.stride
    needle_window = args.needle_window if args.needle_window and args.needle_window > 0 else args.window
    anatomy_window = args.anatomy_window if args.anatomy_window and args.anatomy_window > 0 else args.window

    needle_min_frames = args.needle_min_frames_for_temporal if args.needle_min_frames_for_temporal and args.needle_min_frames_for_temporal > 0 else args.min_frames_for_temporal
    anatomy_min_frames = args.anatomy_min_frames_for_temporal if args.anatomy_min_frames_for_temporal and args.anatomy_min_frames_for_temporal > 0 else args.min_frames_for_temporal

    # splits.json
    all_videos = sorted({r["video_id"] for r in rows})
    train_videos = sorted([v for v in all_videos if v not in val_videos and v not in test_videos])
    splits = {
        "train_videos": train_videos,
        "val_videos": sorted(list(val_videos)),
        "test_videos": sorted(list(test_videos)),
        "all_videos": all_videos,
        "note": "Splits are by video_id. Filenames are parsed from '<video>_frame_<frame>.<ext>'. XML names may include 'images/' prefix.",
        "temporal": {
            "default_window": args.window,
            "needle_window": needle_window,
            "anatomy_window": anatomy_window,
            "stride": stride,
            "default_min_frames": args.min_frames_for_temporal,
            "needle_min_frames": needle_min_frames,
            "anatomy_min_frames": anatomy_min_frames,
        },
    }
    splits_path = meta_dir / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2))
    print("[DONE] Wrote", splits_path)

    # per-video summary (needle presence)
    per_vid = {}
    for r in rows:
        v = r["video_id"]
        per_vid.setdefault(v, {"frames": 0, "needle_pos": 0})
        per_vid[v]["frames"] += 1
        per_vid[v]["needle_pos"] += r["has_needle"]

    print("[INFO] Per-video summary:")
    for v in sorted(per_vid):
        print(f"  video {v}: frames={per_vid[v]['frames']}, needle_pos={per_vid[v]['needle_pos']}")

    # Build sequences
    rows_train = [r for r in rows if r["split"] == "train"]
    rows_val = [r for r in rows if r["split"] == "val"]
    rows_test = [r for r in rows if r["split"] == "test"]

    # anatomy sequences (all targets)
    seq_anat_train = build_sequences(rows_train, "train", anatomy_window, stride, "anatomy", needle_pos_only=False, min_frames=anatomy_min_frames)
    seq_anat_val = build_sequences(rows_val, "val", anatomy_window, stride, "anatomy", needle_pos_only=False, min_frames=anatomy_min_frames)
    seq_anat_test = build_sequences(rows_test, "test", anatomy_window, stride, "anatomy", needle_pos_only=False, min_frames=anatomy_min_frames)

    # needle sequences (only needle-positive targets)
    seq_need_train = build_sequences(rows_train, "train", needle_window, stride, "needle", needle_pos_only=True, min_frames=needle_min_frames)
    seq_need_val = build_sequences(rows_val, "val", needle_window, stride, "needle", needle_pos_only=True, min_frames=needle_min_frames)
    seq_need_test = build_sequences(rows_test, "test", needle_window, stride, "needle", needle_pos_only=True, min_frames=needle_min_frames)

    dump_json(meta_dir / "sequences_anatomy_train.json", seq_anat_train)
    dump_json(meta_dir / "sequences_anatomy_val.json", seq_anat_val)
    dump_json(meta_dir / "sequences_anatomy_test.json", seq_anat_test)

    dump_json(meta_dir / "sequences_needle_train.json", seq_need_train)
    dump_json(meta_dir / "sequences_needle_val.json", seq_need_val)
    dump_json(meta_dir / "sequences_needle_test.json", seq_need_test)

    # Sanity overlays
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # needle overlays: first N positives
    needle_pos_rows = [r for r in rows if r["has_needle"] == 1]
    for i, r in enumerate(needle_pos_rows[: args.overlay_samples]):
        m = np.array(Image.open(r["needle_mask_path"]).convert("L"))
        out = overlay_dir / f"needle_overlay_{i:02d}_v{r['video_id']}_f{r['frame_id']}.png"
        make_overlay(Path(r["image_path"]), m, out, mode="needle")

    # anatomy overlays: first N frames
    for i, r in enumerate(rows[: args.overlay_samples]):
        m = np.array(Image.open(r["anatomy_mask_path"]).convert("L"))
        out = overlay_dir / f"anatomy_overlay_{i:02d}_v{r['video_id']}_f{r['frame_id']}.png"
        make_overlay(Path(r["image_path"]), m, out, mode="anatomy")

    print("[DONE] Sanity overlays saved to", overlay_dir.resolve())
    print("[NEXT] Open sanity_overlays/*.png and verify masks align (and that masks are not all black).")
    print(f"[INFO] needle_window={needle_window} anatomy_window={anatomy_window} stride={stride}")


if __name__ == "__main__":
    main()
