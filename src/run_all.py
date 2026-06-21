import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from train import train_model
from inference import run_inference
from cli import parse_args

if __name__ == '__main__':

    args = parse_args()

    # === Derived configuration (dynamic, dataset-agnostic) ===
    IMG_SIZE = (args.img_size, args.img_size)
    MODEL_PATH = os.path.join(".", "models", f"cae_{args.dataset_name}.h5")

    # === Step 1: Train or load the model ===
    if not args.skip_training:
        print("[INFO] Starting training...")
        model, history = train_model(
            args.train_dir, IMG_SIZE, args.batch_size, args.epochs, MODEL_PATH
        )

        # === Step 2: Plot and save training loss curve ===
        loss_plot_path = os.path.join(args.output_dir, f"{args.dataset_name}_training_loss.png")
        os.makedirs(args.output_dir, exist_ok=True)

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
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model loaded successfully.")

    # === Step 3: Run inference & save results ===
    print("[INFO] Running inference...")
    results = run_inference(
        model, args.test_dir, args.output_dir, IMG_SIZE,
        samples_per_class=args.samples_per_class
    )

    # === Step 4: Plot and save 2x2 grids for each result ===
    print("[INFO] Creating visual result grids...")

    visuals_dir = os.path.join(args.output_dir, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    for item in results:
        input_img   = item['input']
        recon_img   = item['recon']
        heatmap     = item['heatmap']
        overlay     = item['overlay']
        defect_type = item['class']
        name        = item['name']

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

        save_path_class = os.path.join(args.output_dir, defect_type, f"{name}_grid.png")
        plt.savefig(save_path_class)

        plt.close()

    print("[INFO] Done. All visualizations saved in:", visuals_dir)

    print("[INFO] Done. All results and plots saved.")