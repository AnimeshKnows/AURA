"""Sweep orchestration over MVTec categories × depths × loss functions."""

import argparse
import itertools
import os
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from results_logger import has_result
from run_all import run_pipeline
from setup_dataset import MVTEC_CATEGORIES

GRID_IMG_SIZE = 256
DEFAULT_IMG_SIZE = 128
DEFAULT_DATASET_ROOT = r"D:\datasets\mvtec_anomaly_detection"


def resolve_img_size(category: str, requested_img_size: int) -> int:
    """Return img_size for a category; always force 256 for 'grid'."""
    if category == "grid":
        if requested_img_size != GRID_IMG_SIZE:
            print(
                f"[SWEEP] Forcing img_size={GRID_IMG_SIZE} for category 'grid' "
                f"(requested {requested_img_size} ignored)."
            )
        else:
            print(f"[SWEEP] Using img_size={GRID_IMG_SIZE} for category 'grid'.")
        return GRID_IMG_SIZE
    return requested_img_size


def parse_sweep_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AURA sweep: categories × bottleneck depths × loss functions."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=DEFAULT_DATASET_ROOT,
        help=f"MVTec AD root (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(MVTEC_CATEGORIES),
        help="Categories to sweep (default: all 15 MVTec AD categories).",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[2],
        help="Bottleneck depths to sweep (default: [2]).",
    )
    parser.add_argument(
        "--loss_fns",
        nargs="+",
        choices=["mse", "l1", "ssim"],
        default=["mse"],
        help="Loss functions to sweep (default: ['mse']).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=(
            f"Square image size for non-grid categories (default: {DEFAULT_IMG_SIZE}). "
            f"Category 'grid' always uses {GRID_IMG_SIZE}."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per combination (default: 20).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size (default: 32).",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Viz samples per defect class (default: 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base output directory for visualizations (default: './outputs').",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="./results/aura_results.csv",
        help="Results CSV path (default: './results/aura_results.csv').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run combinations even if a matching CSV row already exists.",
    )
    return parser.parse_args(argv)


def run_sweep(args: argparse.Namespace) -> dict:
    combinations = list(
        itertools.product(args.categories, args.depths, args.loss_fns)
    )
    total = len(combinations)
    succeeded = []
    failed = []
    skipped = []

    print(
        f"[SWEEP] Starting sweep: {total} combinations "
        f"({len(args.categories)} categories × {len(args.depths)} depths × "
        f"{len(args.loss_fns)} losses)"
    )

    for index, (category, depth, loss_fn) in enumerate(combinations, start=1):
        img_size = resolve_img_size(category, args.img_size)
        label = f"{category} depth={depth} loss={loss_fn} img_size={img_size}"
        prefix = f"[{index}/{total}] {label}"

        if not args.force and has_result(
            args.results_csv, category, depth, loss_fn, img_size
        ):
            print(f"{prefix}... SKIPPED (already in results CSV)")
            skipped.append(label)
            continue

        train_dir = os.path.join(args.dataset_root, category, "train", "good")
        test_dir = os.path.join(args.dataset_root, category, "test")

        print(f"{prefix}...")
        t0 = time.time()
        try:
            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Missing train/good directory: {train_dir}")
            if not os.path.isdir(test_dir):
                raise FileNotFoundError(f"Missing test directory: {test_dir}")

            result = run_pipeline(
                train_dir=train_dir,
                test_dir=test_dir,
                dataset_name=category,
                output_dir=args.output_dir,
                results_csv=args.results_csv,
                img_size=img_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                bottleneck_depth=depth,
                loss_fn=loss_fn,
                samples_per_class=args.samples_per_class,
                skip_training=False,
            )
            elapsed = time.time() - t0
            auroc = result.get("auroc")
            auroc_str = "N/A" if auroc is None else f"{auroc:.4f}"
            print(f"{prefix}... done, AUROC={auroc_str} ({elapsed:.1f}s)")
            succeeded.append(label)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"{prefix}... FAILED after {elapsed:.1f}s: {exc}")
            traceback.print_exc()
            failed.append((label, str(exc)))

    print()
    print("=" * 60)
    print("[SWEEP] Summary")
    print(f"  Attempted : {total}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("  Failures:")
        for label, err in failed:
            print(f"    - {label}: {err}")
    print("=" * 60)

    return {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
    }


def main(argv=None) -> dict:
    args = parse_sweep_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    summary = main()
    if summary["failed"]:
        sys.exit(1)
    sys.exit(0)
