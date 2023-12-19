# watermarking_functions.py

import numpy as np
import hashlib
import random
import secrets

# Function to embed a watermark into the model using LSB technique
def embed_watermark_LSB(model, watermark_data):
    """
    Embeds a watermark into the provided model using Least Significant Bit (LSB) technique.
    Arguments:
    model : object
        The machine learning model object (e.g., TensorFlow/Keras model).
    watermark_data : str
        The watermark data to be embedded into the model.
    Returns:
    model : object
        The model with the embedded watermark.
    """

    # Convert watermark data to bytes
    watermark_bytes = watermark_data.encode('utf-8')

    # Ensure the watermark is within the capacity of the model parameters
    total_capacity = sum([np.prod(w.shape) for w in model.get_weights()])
    required_capacity = len(watermark_bytes) * 8  # 8 bits per byte
    if required_capacity > total_capacity:
        raise ValueError("Watermark size exceeds model capacity")

    # Flatten and concatenate all model parameters
    flattened_weights = np.concatenate([w.flatten() for w in model.get_weights()])

    # Embed watermark bits into the least significant bits of model parameters
    watermark_bits = ''.join(format(byte, '08b') for byte in watermark_bytes)
    watermark_bits += '1'  # Adding stop bit
    for i, bit in enumerate(watermark_bits):
        flattened_weights[i] = (flattened_weights[i] & ~1) | int(bit)

    # Reshape and update model parameters with embedded watermark
    updated_weights = np.split(flattened_weights, [np.prod(w.shape) for w in model.get_weights()])
    model.set_weights([w.reshape(s) for w, s in zip(updated_weights, [w.shape for w in model.get_weights()])])

    return model

# Function to detect and extract the watermark from the model using LSB detection
def detect_watermark_LSB(model):
    """
    Detects and extracts the watermark from the provided model using Least Significant Bit (LSB) technique.
    Arguments:
    model : object
        The machine learning model object (e.g., TensorFlow/Keras model).
    Returns:
    detected_watermark : str or None
        Extracted watermark if detected, else None.
    """

    # Flatten and concatenate all model parameters
    flattened_weights = np.concatenate([w.flatten() for w in model.get_weights()])

    # Extract watermark bits from the least significant bits of model parameters
    watermark_bits = ''
    stop_bit = '1'
    for bit in flattened_weights:
        bit = int(bit) & 1
        watermark_bits += str(bit)
        if watermark_bits.endswith(stop_bit):
            break

    # Convert extracted bits to bytes and decode watermark
    watermark_bytes = [int(watermark_bits[i:i+8], 2) for i in range(0, len(watermark_bits), 8)]
    detected_watermark = bytearray(watermark_bytes).decode('utf-8')

    return detected_watermark
