from __future__ import annotations

import tensorflow as tf


def build_mlp(width: int, height: int, n_classes: int = 6) -> tf.keras.Model:
    """MLP recebe imagem H x W x C e achata os pixels em um vetor."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="mlp_trashnet",
    )
    return model


def build_cnn(width: int, height: int, n_classes: int = 6) -> tf.keras.Model:
    """CNN com três blocos convolucionais e Global Average Pooling."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(height, width, 3)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="cnn_trashnet",
    )
    return model
