# model_training.py

import tensorflow as tf
from watermarking_functions import embed_watermark_LSB

# Sample data for text classification (replace with your data)
texts = [
    "This is a positive statement.",
    "I love working on machine learning projects.",
    # Add more texts for training
]

# Labels: 0 - Negative sentiment, 1 - Positive sentiment
labels = [1, 1]  # Sample labels (binary classification)

# Tokenizing and preparing the data
max_words = 1000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = tf.keras.preprocessing.sequence.pad_sequences(sequences)

# Define the model architecture (simple example)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, batch_size=32)

# Save the trained model
model.save('text_classification_model.h5')

# Embed watermark into the trained model
watermark_data = "MyWatermark"  # Replace with your watermark data
model_with_watermark = embed_watermark_LSB(model, watermark_data)
model_with_watermark.save('text_classification_model_with_watermark.h5')
