import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime, timezone
from train import train_model
from inference import run_inference
from cli import parse_args, build_model_path, build_run_output_dir
from results_logger import log_result
from losses import get_loss


def run_pipeline(
    train_dir,
    test_dir,
    dataset_name,
    output_dir="./outputs",
    results_csv="./results/aura_results.csv",
    img_size=128,
    epochs=20,
    batch_size=32,
    bottleneck_depth=2,
    loss_fn="mse",
    samples_per_class=3,
    skip_training=False,
):
    """Run train (optional) -> inference/AUROC -> CSV log -> visualization grids.

    ``output_dir`` is treated as a *base* directory; per-run visuals are written
    under a depth/loss-qualified subdirectory to avoid collisions.
    """
    IMG_SIZE = (img_size, img_size)
    MODEL_PATH = build_model_path(
        dataset_name,
        bottleneck_depth=bottleneck_depth,
        loss_fn=loss_fn,
    )
    run_output_dir = build_run_output_dir(
        output_dir, dataset_name, bottleneck_depth, loss_fn
    )

    # === Step 1: Train or load the model ===
    if not skip_training:
        print("[INFO] Starting training...")
        model, history = train_model(
            train_dir,
            IMG_SIZE,
            batch_size,
            epochs,
            MODEL_PATH,
            bottleneck_depth=bottleneck_depth,
            loss_fn=loss_fn,
        )

        # === Step 2: Plot and save training loss curve ===
        loss_plot_path = os.path.join(
            run_output_dir, f"{dataset_name}_training_loss.png"
        )
        os.makedirs(run_output_dir, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CAE Training Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"[INFO] Training loss plot saved to: {loss_plot_path}")

    else:
        print(f"[INFO] --skip_training set. Loading pre-trained model from: {MODEL_PATH}")
        # Custom losses (e.g. ssim) must be registered when loading from disk.
        model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={loss_fn: get_loss(loss_fn)}
            if loss_fn == "ssim"
            else None,
        )
        print("[INFO] Model loaded successfully.")

    # === Step 3: Run inference & save results ===
    print("[INFO] Running inference...")
    results, auroc = run_inference(
        model, test_dir, run_output_dir, IMG_SIZE,
        samples_per_class=samples_per_class,
    )

    # test_dir is one MVTec category's test folder; log one row for that category.
    log_result(
        results_csv,
        {
            "dataset_name": dataset_name,
            "bottleneck_depth": bottleneck_depth,
            "loss_fn": loss_fn,
            "img_size": img_size,
            "category": dataset_name,
            "auroc": "" if auroc is None else f"{auroc:.6f}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[INFO] Logged results to: {results_csv}")
    if auroc is None:
        print("[INFO] AUROC: N/A")
    else:
        print(f"[INFO] AUROC: {auroc:.4f}")

    # === Step 4: Plot and save 2x2 grids for each result ===
    print("[INFO] Creating visual result grids...")

    visuals_dir = os.path.join(run_output_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    for item in results:
        input_img = item['input']
        recon_img = item['recon']
        heatmap = item['heatmap']
        overlay = item['overlay']
        defect_type = item['class']
        name = item['name']

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].imshow(input_img)
        axs[0, 0].set_title("Original")
        axs[0, 1].imshow(recon_img)
        axs[0, 1].set_title("Reconstruction")
        axs[1, 0].imshow(heatmap, cmap='hot')
        axs[1, 0].set_title("Anomaly Heatmap")
        axs[1, 1].imshow(overlay)
        axs[1, 1].set_title("Overlay")

        for ax in axs.ravel():
            ax.axis('off')

        plt.suptitle(f"{defect_type.upper()} - {name}", fontsize=14)
        plt.tight_layout()

        # Both saves must precede plt.close() — bug fix applied here
        save_path_visual = os.path.join(visuals_dir, f"{defect_type}_{name}_grid.png")
        plt.savefig(save_path_visual)

        save_path_class = os.path.join(run_output_dir, defect_type, f"{name}_grid.png")
        plt.savefig(save_path_class)

        plt.close()

    print("[INFO] Done. All visualizations saved in:", visuals_dir)
    print("[INFO] Done. All results and plots saved.")
    return {
        "auroc": auroc,
        "model_path": MODEL_PATH,
        "output_dir": run_output_dir,
    }


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bottleneck_depth=args.bottleneck_depth,
        loss_fn=args.loss_fn,
        samples_per_class=args.samples_per_class,
        skip_training=args.skip_training,
    )
