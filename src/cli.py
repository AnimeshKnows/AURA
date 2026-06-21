import argparse


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