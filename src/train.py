#train.py
import os
import cv2
import numpy as np
import tensorflow as tf
from cae import build_cae
from sklearn.model_selection import train_test_split

def load_images(data_path, img_size):
    images = []
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        images.append(img)
    return np.array(images)

def train_model(data_dir, img_size, batch_size, epochs, model_save_path):
    print("[TRAIN] Loading training data...")
    X = load_images(data_dir, img_size)
    
    # Add channel dimension if needed
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)
    
    input_shape = X.shape[1:]
    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

    print("[TRAIN] Building model...")
    model = build_cae(input_shape)
    model.compile(optimizer='adam', loss='mse')

    print("[TRAIN] Training model...")
    history = model.fit(X_train, X_train,
                        validation_data=(X_val, X_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"[TRAIN] Model saved to {model_save_path}")

    return model, history