import argparse
import os
from typing import Any


def build_model_path(dataset_name: str, **config_kwargs: Any) -> str:
    """Build a model checkpoint path from dataset name and optional config.

    Incorporates ``bottleneck_depth`` and ``loss_fn`` when provided (e.g.
    ``cae_bottle_depth2_mse.h5``). Extra keyword arguments remain accepted so
    future phases can extend naming without changing call sites.
    """
    name = f"cae_{dataset_name}"
    if "bottleneck_depth" in config_kwargs and config_kwargs["bottleneck_depth"] is not None:
        name += f"_depth{config_kwargs['bottleneck_depth']}"
    if "loss_fn" in config_kwargs and config_kwargs["loss_fn"] is not None:
        name += f"_{config_kwargs['loss_fn']}"
    return os.path.join(".", "models", f"{name}.h5")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments for the AURA pipeline."""
    parser = argparse.ArgumentParser(
        description="AURA — Anomaly Understanding & Reconstruction Architecture pipeline."
    )

    # --- Required paths ---
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to the training directory containing defect-free ('good') images."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Path to the test directory containing per-class subdirectories."
    )

    # --- Dataset identity ---
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bottle",
        help="Dataset name used to derive model filename and output subdirectory (default: 'bottle')."
    )

    # --- Output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Root directory for all inference outputs and visualizations (default: './outputs')."
    )

    # --- Model hyperparameters ---
    parser.add_argument(
        "--img_size",
        type=int,
        default=128,
        help="Square image dimension in pixels; produces (img_size x img_size) tensors (default: 128)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Mini-batch size for training (default: 32)."
    )
    parser.add_argument(
        "--bottleneck_depth",
        type=int,
        default=2,
        help=(
            "Number of encoder/decoder conv+pool (or upsample) blocks "
            "(default: 2)."
        ),
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        choices=["mse", "l1", "ssim"],
        default="mse",
        help=(
            "Training loss: 'mse' (mean squared error), 'l1' (mean absolute "
            "error), or 'ssim' (1 - structural similarity). Default: 'mse'."
        ),
    )

    # --- Inference ---
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=3,
        help="Number of test images to sample per defect class during inference (default: 3)."
    )

    # --- Execution mode ---
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="If set, bypass training and load a pre-trained model from the derived MODEL_PATH."
    )

    return parser.parse_args()