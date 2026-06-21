#inference.py
import os
import cv2
import numpy as np
def load_and_preprocess(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    return img
def run_inference(model, test_dir, output_dir, img_size, samples_per_class=3):
    print("[INFER] Running inference...")
    os.makedirs(output_dir, exist_ok=True)
    result_list = []
    for defect_type in sorted(os.listdir(test_dir)):
        class_dir = os.path.join(test_dir, defect_type)
        output_class_dir = os.path.join(output_dir, defect_type)
        os.makedirs(output_class_dir, exist_ok=True)
        img_names = sorted(os.listdir(class_dir)[:samples_per_class])  # Only a few samples
        for img_name in img_names:
            img_path = os.path.join(class_dir, img_name)
            img = load_and_preprocess(img_path, img_size)
            if img is None:
                continue
            input_img = np.expand_dims(img, axis=0)
            recon_img = model.predict(input_img)[0]
            # Error map & heatmap
            error_map = np.abs(img - recon_img)
            heatmap = np.mean(error_map, axis=-1).astype(np.float32)
            if np.max(heatmap) != np.min(heatmap):
                heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
            else:
                heatmap_norm = np.zeros_like(heatmap, dtype=np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted((img * 255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
            base = os.path.splitext(img_name)[0]
            cv2.imwrite(os.path.join(output_class_dir, f"{base}_input.png"), (img * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_class_dir, f"{base}_recon.png"), (recon_img * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(output_class_dir, f"{base}_heatmap.png"), heatmap_colored)
            cv2.imwrite(os.path.join(output_class_dir, f"{base}_overlay.png"), overlay)
            result_list.append({
                'input': img,
                'recon': recon_img,
                'heatmap': heatmap,
                'overlay': overlay,
                'class': defect_type,
                'name': base
            })
    print(f"[INFER] Inference complete. Results saved to '{output_dir}'.")
    return result_list