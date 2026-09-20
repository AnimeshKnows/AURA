"""Verify a local MVTec AD dataset layout without modifying any files."""

import argparse
import os
import sys

MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a local MVTec AD dataset layout (read-only)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"D:\datasets\mvtec_anomaly_detection",
        help=(
            "Root directory containing MVTec AD category folders "
            "(default: D:\\datasets\\mvtec_anomaly_detection)."
        ),
    )
    return parser.parse_args()


def _count_images(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    count = 0
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            count += 1
    return count


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def verify_dataset(dataset_root: str) -> None:
    if not os.path.isdir(dataset_root):
        _fail(f"Dataset root does not exist or is not a directory: {dataset_root}")

    print(f"Verifying MVTec AD layout at: {dataset_root}")
    print(f"Expected categories: {len(MVTEC_CATEGORIES)}")
    print("-" * 60)

    for category in MVTEC_CATEGORIES:
        category_dir = os.path.join(dataset_root, category)
        if not os.path.isdir(category_dir):
            _fail(f"Missing category folder: {category_dir}")

        train_good_dir = os.path.join(category_dir, "train", "good")
        if not os.path.isdir(train_good_dir):
            _fail(f"Missing train/good folder for '{category}': {train_good_dir}")

        test_dir = os.path.join(category_dir, "test")
        if not os.path.isdir(test_dir):
            _fail(f"Missing test folder for '{category}': {test_dir}")

        train_good_count = _count_images(train_good_dir)
        if train_good_count == 0:
            _fail(f"No training images found in: {train_good_dir}")

        test_subdirs = sorted(
            name
            for name in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, name))
        )
        if not test_subdirs:
            _fail(f"No test subclasses found under: {test_dir}")

        print(f"[{category}]")
        print(f"  train/good: {train_good_count} images")
        print("  test:")
        for subclass in test_subdirs:
            subclass_dir = os.path.join(test_dir, subclass)
            n = _count_images(subclass_dir)
            if n == 0:
                _fail(f"No test images found in: {subclass_dir}")
            print(f"    {subclass}: {n} images")
        print()

    print("-" * 60)
    print(f"OK: All {len(MVTEC_CATEGORIES)} MVTec AD categories verified.")


def main() -> None:
    args = parse_args()
    verify_dataset(args.dataset_root)


if __name__ == "__main__":
    main()
