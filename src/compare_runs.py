# src/compare_runs.py
"""Compare multiple training runs side-by-side.

Reads metrics.csv and summary.json from each run_* subfolder under a given
root directory, then prints a ranked table sorted by best validation Dice.
Works for both anatomy and needle run folders.

Usage:
    python src/compare_runs.py                              # default: anatomy
    python src/compare_runs.py --root runs/needle_convlstm  # needle runs
    python src/compare_runs.py --root runs/anatomy_swin_convlstm --sort fp_rate_nerve
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_metrics_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _best_epoch_row(rows: list[dict], metric_key: str, higher_better: bool = True) -> dict | None:
    if not rows:
        return None
    best = None
    for r in rows:
        val = r.get(metric_key)
        if val is None or val == "":
            continue
        val = float(val)
        if best is None:
            best = r
        else:
            if higher_better and val > float(best[metric_key]):
                best = r
            elif (not higher_better) and val < float(best[metric_key]):
                best = r
    return best


def collect_runs(root: Path) -> list[dict]:
    runs = []
    if not root.exists():
        return runs

    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue

        summary = _read_json(d / "summary.json")
        run_cfg = _read_json(d / "run_config.json")
        metrics = _read_metrics_csv(d / "metrics.csv")

        entry: dict = {"run_dir": d.name, "path": str(d)}

        # ---- extract args from run_config ----
        args = {}
        if isinstance(run_cfg, dict):
            args = run_cfg.get("args", {})
            if not isinstance(args, dict):
                args = {}

        entry["arch"] = str(args.get("arch", ""))
        entry["loss"] = str(args.get("loss", ""))
        entry["T"] = str(args.get("T", ""))
        entry["class_weights"] = str(args.get("class_weights", ""))
        entry["lambda_dice"] = str(args.get("lambda_dice", ""))
        entry["lambda_ce"] = str(args.get("lambda_ce", ""))
        entry["lr"] = str(args.get("lr", ""))
        entry["epochs_total"] = str(args.get("epochs", ""))
        entry["scheduler"] = str(args.get("scheduler", ""))

        # ---- best metrics from summary.json ----
        if isinstance(summary, dict):
            entry["best_epoch"] = str(summary.get("best_epoch", ""))
            # anatomy key
            if "best_val_mean_dice_ex_bg" in summary:
                entry["best_dice"] = f"{float(summary['best_val_mean_dice_ex_bg']):.4f}"
            # needle key
            elif "best_val_dice" in summary:
                entry["best_dice"] = f"{float(summary['best_val_dice']):.4f}"
            else:
                entry["best_dice"] = ""
        else:
            entry["best_epoch"] = ""
            entry["best_dice"] = ""

        # ---- per-class detail from best epoch row in metrics.csv ----
        if metrics:
            # Try anatomy-style first
            best_row = _best_epoch_row(metrics, "val_mean_dice_ex_bg", higher_better=True)
            if best_row is None:
                best_row = _best_epoch_row(metrics, "val_dice", higher_better=True)

            _metric_keys = [
                "dice_nerve", "dice_artery", "dice_muscle",
                "iou_nerve", "iou_artery", "iou_muscle", "val_mean_iou_ex_bg",
                "prec_nerve", "prec_artery", "prec_muscle",
                "rec_nerve", "rec_artery", "rec_muscle",
                "fp_rate_nerve", "fp_rate_artery", "fp_rate_muscle",
                "val_dice", "train_loss",
            ]

            if best_row:
                for key in _metric_keys:
                    val = best_row.get(key)
                    if val is not None and val != "":
                        entry[key] = f"{float(val):.4f}"
                    else:
                        entry[key] = ""

                # Last epoch train loss (for convergence check)
                last_row = metrics[-1]
                entry["final_train_loss"] = f"{float(last_row.get('train_loss', 0)):.4f}" if last_row.get("train_loss") else ""
            else:
                for key in _metric_keys + ["final_train_loss"]:
                    entry[key] = ""
        else:
            _metric_keys = [
                "dice_nerve", "dice_artery", "dice_muscle",
                "iou_nerve", "iou_artery", "iou_muscle", "val_mean_iou_ex_bg",
                "prec_nerve", "prec_artery", "prec_muscle",
                "rec_nerve", "rec_artery", "rec_muscle",
                "fp_rate_nerve", "fp_rate_artery", "fp_rate_muscle",
                "val_dice", "train_loss", "final_train_loss",
            ]
            for key in _metric_keys:
                entry[key] = ""

        runs.append(entry)

    return runs


def _sort_key(entry: dict, sort_col: str, higher_better: bool) -> float:
    val = entry.get(sort_col, "")
    if val == "" or val is None:
        return float("-inf") if higher_better else float("inf")
    try:
        v = float(val)
        return v if higher_better else -v
    except ValueError:
        return float("-inf") if higher_better else float("inf")


def print_table(runs: list[dict], task: str, sort_col: str):
    if not runs:
        print("[INFO] No runs found.")
        return

    # Choose columns based on task
    _higher_cols = {"best_dice", "dice_nerve", "dice_artery", "dice_muscle", "val_dice",
                     "iou_nerve", "iou_artery", "iou_muscle", "val_mean_iou_ex_bg",
                     "prec_nerve", "prec_artery", "prec_muscle",
                     "rec_nerve", "rec_artery", "rec_muscle"}
    higher_better = sort_col in _higher_cols

    runs_sorted = sorted(runs, key=lambda e: _sort_key(e, sort_col, higher_better), reverse=higher_better)

    if task == "anatomy":
        columns = [
            ("Rank", 4),
            ("run_dir", 38),
            ("loss", 8),
            ("sched", 6),
            ("cls_wt", 6),
            ("ep", 4),
            ("dice", 7),
            ("iou", 7),
            ("d_nrv", 6),
            ("d_art", 6),
            ("d_mus", 6),
            ("p_nrv", 6),
            ("r_nrv", 6),
            ("fp_nrv", 7),
            ("fp_art", 7),
            ("loss_f", 7),
        ]
        key_map = {
            "Rank": None,
            "run_dir": "run_dir",
            "loss": "loss",
            "sched": "scheduler",
            "cls_wt": "class_weights",
            "ep": "best_epoch",
            "dice": "best_dice",
            "iou": "val_mean_iou_ex_bg",
            "d_nrv": "dice_nerve",
            "d_art": "dice_artery",
            "d_mus": "dice_muscle",
            "p_nrv": "prec_nerve",
            "r_nrv": "rec_nerve",
            "fp_nrv": "fp_rate_nerve",
            "fp_art": "fp_rate_artery",
            "loss_f": "final_train_loss",
        }
    else:
        columns = [
            ("Rank", 4),
            ("run_dir", 38),
            ("best_epoch", 6),
            ("best_dice", 10),
            ("final_loss", 10),
            ("T", 3),
            ("lr", 8),
        ]
        key_map = {
            "Rank": None,
            "run_dir": "run_dir",
            "best_epoch": "best_epoch",
            "best_dice": "best_dice",
            "final_loss": "final_train_loss",
            "T": "T",
            "lr": "lr",
        }

    # Header
    header = "  ".join(col.ljust(w) for col, w in columns)
    print(f"\n{'=' * len(header)}")
    print(f" {task.upper()} RUNS — sorted by {sort_col} ({'higher' if higher_better else 'lower'} is better)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for rank, entry in enumerate(runs_sorted, start=1):
        parts = []
        for col, w in columns:
            data_key = key_map.get(col)
            if col == "Rank":
                parts.append(str(rank).ljust(w))
            elif data_key:
                parts.append(str(entry.get(data_key, "")).ljust(w))
            else:
                parts.append("".ljust(w))
        print("  ".join(parts))

    print(f"{'=' * len(header)}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Compare training runs side-by-side, ranked by validation Dice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--root",
        default="runs/anatomy_swin_convlstm",
        help="Root folder containing run_* subfolders to compare.",
    )
    ap.add_argument(
        "--sort",
        default="best_dice",
        help="Column to sort by. Options: best_dice, dice_nerve, dice_artery, dice_muscle, fp_rate_nerve, fp_rate_artery, fp_rate_muscle, final_train_loss.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERROR] Root folder not found: {root.resolve()}")
        return

    # Detect task from folder name
    task = "anatomy" if "anatomy" in root.name else "needle"
    runs = collect_runs(root)
    print_table(runs, task, args.sort)


if __name__ == "__main__":
    main()
