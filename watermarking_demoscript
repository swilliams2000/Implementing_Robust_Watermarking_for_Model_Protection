# demo_script.py

import tensorflow as tf
from watermarking_functions import detect_watermark_LSB

# Load the trained model with the embedded watermark
model_with_watermark = tf.keras.models.load_model('text_classification_model_with_watermark.h5')

# Detect and extract the watermark from the model
detected_watermark = detect_watermark_LSB(model_with_watermark)

if detected_watermark:
    print("Watermark Detected:", detected_watermark)
else:
    print("No watermark found or watermark detection failed.")
