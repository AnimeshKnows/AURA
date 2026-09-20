"""Loss function registry for CAE training."""

import tensorflow as tf


def ssim_loss(y_true, y_pred):
    """1 - mean SSIM. Images are normalized to [0, 1], so max_val=1.0."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def get_loss(name: str):
    """Return a Keras-compatible loss for ``name`` in {'mse', 'l1', 'ssim'}."""
    losses = {
        "mse": "mse",
        "l1": "mae",  # selectable name is 'l1'; Keras built-in is mae
        "ssim": ssim_loss,
    }
    if name not in losses:
        raise ValueError(
            f"Unknown loss_fn '{name}'. Expected one of: {sorted(losses)}"
        )
    return losses[name]
