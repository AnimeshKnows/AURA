# cae.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D


def build_cae(input_shape=(64, 64, 3), depth=2, base_filters=32):
    """Build a convolutional autoencoder with a configurable encoder/decoder depth.

    Filter schedule matches the historical hardcoded model at depth=2,
    base_filters=32 (encoder 32→16, decoder 16→32, then Conv2D(3, sigmoid)).
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    height, width = int(input_shape[0]), int(input_shape[1])
    min_side = min(height, width)
    # Each MaxPooling2D(2) halves spatial size; after `depth` pools we need >= 1x1.
    max_depth = 0
    side = min_side
    while side >= 2:
        side //= 2
        max_depth += 1
    if depth > max_depth:
        raise ValueError(
            f"depth={depth} would reduce spatial size of {height}x{width} input "
            f"below 1x1 (maximum depth for this input is {max_depth})"
        )

    encoder_filters = []
    for i in range(depth):
        filters = base_filters // (2 ** i)
        if filters < 1:
            raise ValueError(
                f"depth={depth} with base_filters={base_filters} yields "
                f"< 1 filters at encoder block {i}"
            )
        encoder_filters.append(filters)

    input_img = Input(shape=input_shape)
    x = input_img

    # Encoder: depth blocks of Conv2D + MaxPooling2D
    for filters in encoder_filters:
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    encoded = x

    # Decoder: mirrored filter schedule with Upsampling after each conv
    x = encoded
    for filters in reversed(encoder_filters):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(input_img, decoded)
