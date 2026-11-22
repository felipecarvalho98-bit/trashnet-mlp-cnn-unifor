"""Leitura de imagens e acompanhamento do treinamento."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from .config import CLASS_NAMES, BATCH_SIZE


def make_dataset(frame, image_size: tuple[int, int], batch_size: int = BATCH_SIZE,
                 training: bool = False, repeat: bool = False):
    "Prepara as imagens para o TensorFlow: lê os arquivos, converte para RGB, redimensiona, normaliza os pixels e separa os exemplos em lotes. O embaralhamento acontece apenas nos dados de treino."
    paths = frame["path"].astype(str).to_numpy()
    labels = frame["class_index"].astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size, antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    if training:
        ds = ds.shuffle(buffer_size=len(frame), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not training)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options).prefetch(1)
    return ds


class TrainingMonitor(tf.keras.callbacks.Callback):
    """Calcula macro-F1 de validação e salva o melhor checkpoint por macro-F1."""
    def __init__(self, validation_ds, y_validation, checkpoint_path: Path, log_path: Path):
        "Guarda os dados de validação e os caminhos onde o melhor modelo e o histórico serão salvos. Também prepara a medição opcional de memória do processo."
        super().__init__()
        self.validation_ds = validation_ds
        self.y_validation = np.asarray(y_validation, dtype=int)
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.best_f1 = -1.0
        self.epoch_start = None
        self.rows = []
        self.process = None
        try:
            import psutil
            self.process = psutil.Process(os.getpid())
        except Exception:
            pass

    def on_epoch_begin(self, epoch, logs=None):
        "Marca o início da época para calcular quanto tempo ela levou."
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        "Ao terminar cada época, calcula o F1 macro na validação, registra métricas e tempo, e salva os pesos quando encontra o melhor resultado até aquele momento."
        logs = logs if logs is not None else {}
        probabilities = self.model.predict(self.validation_ds, verbose=0)
        predicted = np.argmax(probabilities, axis=1)
        macro_f1 = float(f1_score(self.y_validation, predicted, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0))
        elapsed = time.perf_counter() - (self.epoch_start or time.perf_counter())
        rss_mb = None
        if self.process is not None:
            try:
                rss_mb = self.process.memory_info().rss / (1024 ** 2)
            except Exception:
                pass
        logs["val_macro_f1"] = macro_f1
        logs["epoch_time_seconds"] = elapsed
        if rss_mb is not None:
            logs["process_rss_mb"] = rss_mb
        row = {"epoch": int(epoch + 1)}
        row.update({k: float(v) for k, v in logs.items() if np.isscalar(v)})
        self.rows.append(row)
        # Salva log a cada época, permitindo recuperar o histórico se a execução for interrompida.
        import pandas as pd
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.model.save_weights(self.checkpoint_path)
            print(f" — melhor val_macro_f1: {macro_f1:.4f}; checkpoint salvo")
