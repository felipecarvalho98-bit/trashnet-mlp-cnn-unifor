from __future__ import annotations

import argparse
import os
import platform
import time
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import psutil
import tensorflow as tf

from src.common import (
    CHECKPOINTS_DIR, CLASSES, EXPERIMENTS, HISTORY_DIR, METRICS_DIR, MODELS_DIR, OUTPUT_DIR,
    SEED, find_dataset_dir, load_json, make_output_dirs, save_json,
    validate_reference_dataset_counts, validate_splits,
)
from src.models import build_cnn, build_mlp

MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8
BATCH_FALLBACKS = [32, 16, 8, 4, 2]
LEARNING_RATE = 1e-3


def configure_tensorflow() -> list:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Aviso: não foi possível ativar memory growth: {exc}")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return gpus


def build_dataset(dataset_dir: Path, relative_paths: list[str], labels: list[int],
                  width: int, height: int, batch_size: int, training: bool) -> tf.data.Dataset:
    absolute_paths = [str(dataset_dir / relative) for relative in relative_paths]
    dataset = tf.data.Dataset.from_tensor_slices((absolute_paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(absolute_paths), seed=SEED, reshuffle_each_iteration=True
        )

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [height, width], method="bilinear", antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(label, tf.int32)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.AUTOTUNE)


class MemoryMonitor(tf.keras.callbacks.Callback):
    """Amostra RAM, alocador do TensorFlow e VRAM do processo via nvidia-smi."""
    def __init__(self, gpu_enabled: bool):
        super().__init__()
        self.process = psutil.Process(os.getpid())
        self.gpu_enabled = gpu_enabled
        self.peak_ram_mb = 0.0
        self.peak_gpu_tensorflow_mb = None
        self.peak_gpu_process_nvidia_smi_mb = None
        self.gpu_utilization_samples = []
        self.device = "GPU:0"
        self._batch_counter = 0

    def _nvidia_smi_process_memory_mb(self):
        if not self.gpu_enabled:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            current_pid = str(os.getpid())
            memory_values = []
            for line in result.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 2 and parts[0] == current_pid:
                    try:
                        memory_values.append(float(parts[1]))
                    except ValueError:
                        continue
            return sum(memory_values) if memory_values else None
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None

    def _nvidia_smi_gpu_utilization(self):
        if not self.gpu_enabled:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=True,
            )
            first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
            if first_line and first_line.lower() not in {"n/a", "[not supported]"}:
                return float(first_line)
        except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
            return None
        return None

    def _sample(self):
        try:
            rss = self.process.memory_info().rss / (1024 ** 2)
            self.peak_ram_mb = max(self.peak_ram_mb, rss)
        except Exception:
            pass

        if self.gpu_enabled:
            try:
                memory = tf.config.experimental.get_memory_info(self.device)
                peak = memory.get("peak")
                if peak is not None:
                    peak_mb = peak / (1024 ** 2)
                    self.peak_gpu_tensorflow_mb = max(
                        self.peak_gpu_tensorflow_mb or 0.0, peak_mb
                    )
            except Exception:
                pass
            nvidia_smi_mb = self._nvidia_smi_process_memory_mb()
            if nvidia_smi_mb is not None:
                self.peak_gpu_process_nvidia_smi_mb = max(
                    self.peak_gpu_process_nvidia_smi_mb or 0.0, nvidia_smi_mb
                )
            gpu_utilization = self._nvidia_smi_gpu_utilization()
            if gpu_utilization is not None:
                self.gpu_utilization_samples.append(gpu_utilization)

    def on_train_batch_end(self, batch, logs=None):
        self._batch_counter += 1
        if self._batch_counter % 5 == 0:
            self._sample()

    def on_test_batch_end(self, batch, logs=None):
        if batch % 5 == 0:
            self._sample()

    def on_epoch_end(self, epoch, logs=None):
        self._sample()


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_seconds: list[float] = []
        self._start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        if self._start is not None:
            elapsed = float(time.perf_counter() - self._start)
            self.epoch_seconds.append(elapsed)
            if logs is not None:
                logs["epoch_seconds"] = elapsed


def build_model(experiment: str, width: int, height: int) -> tf.keras.Model:
    model_type = EXPERIMENTS[experiment]["model"]
    if model_type == "mlp":
        model = build_mlp(width, height, len(CLASSES))
    else:
        model = build_cnn(width, height, len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def get_hardware_info(gpus: list) -> dict:
    gpu_names = []
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            gpu_names.append(details.get("device_name", gpu.name))
        except Exception:
            gpu_names.append(gpu.name)
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "gpu_detected": bool(gpus),
        "gpu_names": gpu_names,
        "gpu_devices_nvidia_smi": _nvidia_smi_gpu_summary(),
        "cpu_count": os.cpu_count(),
    }



def _read_history_csv(csv_path: Path) -> dict[str, list[float]]:
    """Lê o histórico completo, inclusive épocas concluídas antes de uma retomada."""
    history: dict[str, list[float]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                if key in (None, "epoch") or value in (None, ""):
                    continue
                try:
                    history.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return history


def _nvidia_smi_gpu_summary() -> list[dict]:
    """Retorna nome e memória total das GPUs, quando nvidia-smi estiver disponível."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=True,
        )
        devices = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                try:
                    devices.append({"name": parts[0], "total_memory_mb": float(parts[1])})
                except ValueError:
                    devices.append({"name": parts[0], "total_memory_mb": None})
        return devices
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []


def train_one(experiment: str, dataset_dir: Path, splits: dict, gpus: list) -> dict:
    cfg = EXPERIMENTS[experiment]
    width, height = cfg["width"], cfg["height"]
    train_paths = splits["paths"]["train"]
    val_paths = splits["paths"]["val"]
    y_train = splits["labels"]["train"]
    y_val = splits["labels"]["val"]

    tf.keras.utils.set_random_seed(SEED)
    class_weights = None
    if cfg["class_weights"]:
        class_weights = {
            index: float(splits["class_weights_from_training_only"][class_name])
            for index, class_name in enumerate(CLASSES)
        }

    model_path = MODELS_DIR / f"modelo_{experiment}.keras"
    history_path = HISTORY_DIR / f"historico_{experiment}.json"
    config_path = METRICS_DIR / f"config_{experiment}.json"

    attempts = []
    successful = None
    backup_dir = CHECKPOINTS_DIR / f"backup_{experiment}"
    csv_log_path = HISTORY_DIR / f"epocas_{experiment}.csv"
    # Se não existe checkpoint de retomada, este é um treinamento novo.
    # Nesse caso, remove um CSV antigo para não misturar épocas de execuções diferentes.
    has_resume_backup = backup_dir.exists() and any(backup_dir.iterdir())
    if not has_resume_backup:
        csv_log_path.unlink(missing_ok=True)

    for batch_size in BATCH_FALLBACKS:
        tf.keras.backend.clear_session()
        model = build_model(experiment, width, height)
        train_ds = build_dataset(dataset_dir, train_paths, y_train, width, height, batch_size, True)
        val_ds = build_dataset(dataset_dir, val_paths, y_val, width, height, batch_size, False)

        gpu_device = "GPU:0"
        if gpus:
            try:
                tf.config.experimental.reset_memory_stats(gpu_device)
            except Exception:
                pass

        memory_monitor = MemoryMonitor(gpu_enabled=bool(gpus))
        epoch_timer = EpochTimer()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, mode="min", verbose=1,
            ),
            epoch_timer,  # injeta epoch_seconds antes do CSVLogger registrar a época
            tf.keras.callbacks.CSVLogger(str(csv_log_path), append=True),
            memory_monitor,
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=str(backup_dir), save_freq="epoch"
            ),
        ]

        start = time.perf_counter()
        try:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=MAX_EPOCHS,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=2,
            )
            elapsed = time.perf_counter() - start
            successful = (model, history, batch_size, elapsed, memory_monitor, epoch_timer)
            attempts.append({"batch_size": batch_size, "status": "success", "seconds": round(elapsed, 3)})
            break
        except tf.errors.ResourceExhaustedError as exc:
            elapsed = time.perf_counter() - start
            attempts.append({
                "batch_size": batch_size,
                "status": "out_of_memory",
                "seconds": round(elapsed, 3),
                "message": str(exc)[:500],
            })
            print(f"Memória insuficiente com lote {batch_size}. Tentando lote menor...")
            del model, train_ds, val_ds
            tf.keras.backend.clear_session()
            # A tentativa seguinte começa do zero, pois mudou o tamanho do lote.
            shutil.rmtree(backup_dir, ignore_errors=True)
            csv_log_path.unlink(missing_ok=True)
            has_resume_backup = False
            if batch_size == BATCH_FALLBACKS[-1]:
                raise

    if successful is None:
        raise RuntimeError(f"Experimento {experiment} terminou sem modelo treinado.")

    model, history, batch_size, elapsed, memory_monitor, epoch_timer = successful
    model.save(model_path)

    history_dict = _read_history_csv(csv_log_path)
    if not history_dict.get("loss"):
        # Fallback para ambientes/versões que não gravem o CSV como esperado.
        history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    epoch_seconds = history_dict.get("epoch_seconds", epoch_timer.epoch_seconds)
    executed_epochs = len(history_dict.get("loss", []))
    best_epoch = int(np.argmin(history_dict["val_loss"]) + 1) if history_dict.get("val_loss") else None
    training_total_seconds = float(sum(epoch_seconds)) if epoch_seconds else float(elapsed)
    history_dict["epochs_executed"] = executed_epochs
    history_dict["best_epoch_by_val_loss"] = best_epoch
    history_dict["epoch_seconds_sampled"] = epoch_seconds
    save_json(history_dict, history_path)

    config = {
        "experiment": experiment,
        "model_type": cfg["model"],
        "input_width_height": [width, height],
        "keras_input_shape_height_width_channels": [height, width, 3],
        "class_weights_enabled": bool(cfg["class_weights"]),
        "class_weights_used": class_weights,
        "batch_size_used": batch_size,
        "batch_size_attempts": attempts,
        "epochs_executed": executed_epochs,
        "best_epoch_by_val_loss": best_epoch,
        "total_parameters": int(model.count_params()),
        "trainable_parameters": int(sum(np.prod(weight.shape) for weight in model.trainable_weights)),
        "model_size_mb": round(model_path.stat().st_size / (1024 ** 2), 3),
        "training_total_seconds": round(training_total_seconds, 3),
        "training_mean_seconds_per_epoch": round(training_total_seconds / max(executed_epochs, 1), 3),
        "training_current_session_fit_seconds": round(elapsed, 3),
        "ram_peak_sampled_mb": round(memory_monitor.peak_ram_mb, 2),
        "gpu_tensorflow_allocator_peak_mb": (
            round(memory_monitor.peak_gpu_tensorflow_mb, 2)
            if memory_monitor.peak_gpu_tensorflow_mb is not None else None
        ),
        "gpu_process_peak_nvidia_smi_mb": (
            round(memory_monitor.peak_gpu_process_nvidia_smi_mb, 2)
            if memory_monitor.peak_gpu_process_nvidia_smi_mb is not None else None
        ),
        "gpu_utilization_mean_sampled_percent": (
            round(float(np.mean(memory_monitor.gpu_utilization_samples)), 2)
            if memory_monitor.gpu_utilization_samples else None
        ),
        "gpu_utilization_max_sampled_percent": (
            round(float(np.max(memory_monitor.gpu_utilization_samples)), 2)
            if memory_monitor.gpu_utilization_samples else None
        ),
        "gpu_utilization_sample_count": len(memory_monitor.gpu_utilization_samples),
        "nvidia_smi_devices": _nvidia_smi_gpu_summary(),
        "hardware": get_hardware_info(gpus),
        "seed": SEED,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "sparse_categorical_crossentropy",
        "early_stopping": {
            "monitor": "val_loss",
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
            "max_epochs": MAX_EPOCHS,
        },
        "timing_method": (
            "time.perf_counter ao redor de model.fit; inclui leitura/decodificação/redimensionamento "
            "por tf.data durante o treinamento; exclui download, inspeção inicial, montagem de "
            "datasets, compilação e gravação do arquivo .keras."
        ),
        "memory_method": (
            "RAM: RSS do processo amostrado em lotes/épocas. GPU: pico do alocador TensorFlow "
            "via tf.config.experimental.get_memory_info, pico de memória do processo via nvidia-smi "
            "e utilização média/máxima amostrada da GPU. A utilização é do dispositivo, não uma "
            "medida exclusiva do processo. Todas são amostradas; null indica dado indisponível."
        ),
    }
    save_json(config, config_path)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinamento dos seis experimentos obrigatórios.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Treina E1 até E6.")
    group.add_argument("--exp", choices=list(EXPERIMENTS), help="Treina apenas um experimento.")
    parser.add_argument("--skip-existing", action="store_true", help="Pula experimentos com modelo e configuração já salvos.")
    args = parser.parse_args()

    make_output_dirs()
    dataset_dir = find_dataset_dir()
    validate_reference_dataset_counts(dataset_dir)
    splits_path = OUTPUT_DIR / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError("Execute primeiro: python -m src.prepare_data")
    splits = load_json(splits_path)
    validate_splits(splits)
    gpus = configure_tensorflow()
    print("Dispositivos GPU detectados:", [gpu.name for gpu in gpus] or "nenhum (CPU)")
    experiments = [args.exp] if args.exp else list(EXPERIMENTS)

    for experiment in experiments:
        model_path = MODELS_DIR / f"modelo_{experiment}.keras"
        config_path = METRICS_DIR / f"config_{experiment}.json"
        if args.skip_existing and model_path.exists() and config_path.exists():
            print(f"{experiment}: modelo e configuração já existem; pulando.")
            continue
        print(f"\n{'=' * 20} INICIANDO {experiment} {'=' * 20}")
        train_one(experiment, dataset_dir, splits, gpus)
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
