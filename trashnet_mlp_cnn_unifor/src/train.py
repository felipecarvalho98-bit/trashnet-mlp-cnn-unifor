"""Treinamento dos experimentos E1–E6."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import (CLASS_NAMES, EXPERIMENTS, OUTPUT_DIR, BATCH_SIZE, MAX_EPOCHS,
                     EARLY_STOPPING_PATIENCE, LEARNING_RATE, SEED, experiment_dir)
from .data_utils import load_splits
from .models import build_model
from .training_utils import make_dataset, TrainingMonitor


def configure_tensorflow() -> None:
    # Ajuste conservador para computador Windows com 16 GB de RAM.
    "Aplica alguns limites de paralelismo para manter o uso de CPU e memória mais controlado. Também ativa operações determinísticas quando a versão do TensorFlow permite."
    try:
        tf.config.threading.set_intra_op_parallelism_threads(max(1, min(4, os.cpu_count() or 2)))
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def hardware_info() -> dict:
    "Registra informações da máquina e do TensorFlow. Esses dados ajudam a interpretar o tempo de treinamento e a identificar se a execução ocorreu em CPU ou GPU."
    import psutil
    try:
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        gpus = []
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "processor": platform.processor() or "Não informado",
        "logical_cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu_devices_detected_by_tensorflow": gpus,
        "execution_device": "GPU" if gpus else "CPU",
    }


def train_one(experiment_id: str, epochs: int, force: bool = False, max_steps: int | None = None) -> None:
    "Executa um experimento completo: prepara os dados, monta a arquitetura, aplica pesos de classe quando previstos, treina, acompanha o F1 macro da validação e salva o melhor modelo junto com seus metadados."
    cfg = EXPERIMENTS[experiment_id]
    # Execuções curtas de diagnóstico são isoladas e nunca ocupam o diretório oficial do experimento.
    if max_steps is not None:
        out_dir = OUTPUT_DIR / "_smoke_test" / f"{experiment_id}_{cfg['name']}"
    else:
        out_dir = experiment_dir(experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "best_model.keras"
    if model_path.exists() and not force and max_steps is None:
        print(f"\n{experiment_id}: modelo existente encontrado; pulando. Use --force para refazer.")
        return

    splits, dataset_root = load_splits()
    train_df = splits[splits.split == "train"].reset_index(drop=True)
    val_df = splits[splits.split == "validation"].reset_index(drop=True)
    y_train = train_df.class_index.to_numpy(dtype=int)
    y_val = val_df.class_index.to_numpy(dtype=int)
    height, width = cfg["image_size"]
    input_shape = (height, width, 3)

    train_ds = make_dataset(train_df, cfg["image_size"], training=True, repeat=max_steps is not None)
    val_ds = make_dataset(val_df, cfg["image_size"], training=False)

    tf.keras.utils.set_random_seed(SEED)
    model = build_model(cfg["model_type"], input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    weights = None
    weight_map = None
    if cfg["class_weights"]:
        class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=y_train)
        weights = {int(i): float(w) for i, w in enumerate(class_weights)}
        weight_map = {CLASS_NAMES[i]: float(weights[i]) for i in range(len(CLASS_NAMES))}

    checkpoint_path = out_dir / "best.weights.h5"
    epoch_log_path = out_dir / "historico_treinamento.csv"
    monitor = TrainingMonitor(val_ds, y_val, checkpoint_path, epoch_log_path)
    callbacks = [
        monitor,
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=False, verbose=1),
    ]
    start = time.perf_counter()
    interrupted = False
    print(f"\n{'=' * 18} INICIANDO {experiment_id} {'=' * 18}")
    print(f"Modelo: {cfg['model_type']} | Resolução: {width}x{height} | Pesos de classe: {cfg['class_weights']}")
    print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Batch: {BATCH_SIZE} | Épocas máximas: {epochs}")
    try:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=weights,
            steps_per_epoch=max_steps,
            verbose=2,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nTreinamento interrompido pelo usuário. Histórico e melhor checkpoint serão preservados se houver época concluída.")
    total_seconds = time.perf_counter() - start
    if not checkpoint_path.exists():
        raise RuntimeError(f"{experiment_id}: nenhum checkpoint foi salvo. Confira os dados e tente novamente.")
    model.load_weights(checkpoint_path)
    model.save(model_path)
    history_df = pd.read_csv(epoch_log_path) if epoch_log_path.exists() else pd.DataFrame()
    best_epoch = int(history_df.loc[history_df.val_macro_f1.idxmax(), "epoch"]) if not history_df.empty else None
    if not history_df.empty:
        history_df.to_json(out_dir / "historico_treinamento.json", orient="records", indent=2)
    with (out_dir / "resumo_modelo.txt").open("w", encoding="utf-8") as summary_file:
        model.summary(print_fn=lambda line: summary_file.write(line + "\n"))
    metadata = {
        "experiment_id": experiment_id,
        "description": cfg["description"],
        "model_type": cfg["model_type"],
        "image_width": width,
        "image_height": height,
        "channels": 3,
        "normalization": "RGB float32 dividido por 255 (0 a 1)",
        "class_weights_used": bool(cfg["class_weights"]),
        "class_weights": weight_map,
        "augmentation": "Não utilizada; configuração idêntica entre partições e experimentos, exceto variáveis definidas no protocolo.",
        "optimizer": "Adam",
        "learning_rate_initial": LEARNING_RATE,
        "loss": "SparseCategoricalCrossentropy",
        "batch_size": BATCH_SIZE,
        "max_epochs": epochs,
        "epochs_completed": int(len(history_df)),
        "best_epoch_by_validation_macro_f1": best_epoch,
        "best_validation_macro_f1": float(history_df.val_macro_f1.max()) if not history_df.empty else None,
        "total_training_seconds": total_seconds,
        "average_epoch_seconds": float(history_df.epoch_time_seconds.mean()) if not history_df.empty and "epoch_time_seconds" in history_df else None,
        "process_ram_peak_mb_sampled": float(history_df.process_rss_mb.max()) if not history_df.empty and "process_rss_mb" in history_df else None,
        "model_parameters": int(model.count_params()),
        "model_file_size_mb": round(model_path.stat().st_size / (1024 ** 2), 3),
        "batch_steps_limit_for_smoke_test": max_steps,
        "interrupted_by_user": interrupted,
        "dataset_root": str(dataset_root),
        "seed": SEED,
        "hardware": hardware_info(),
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{experiment_id} concluído. Melhor época pela macro-F1 de validação: {best_epoch}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Tempo total: {total_seconds / 60:.1f} min")


def main() -> None:
    "Lê os argumentos do terminal e define se será executado um experimento específico ou a sequência E1–E6. Também valida os limites de épocas e passos usados em testes rápidos."
    parser = argparse.ArgumentParser(description="Treinamento dos seis experimentos TrashNet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Executa E1 até E6")
    group.add_argument("--experiment", choices=list(EXPERIMENTS), help="Executa somente um experimento")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Máximo de épocas (padrão: {MAX_EPOCHS})")
    parser.add_argument("--force", action="store_true", help="Refaz mesmo se já houver modelo salvo")
    parser.add_argument("--max-steps", type=int, default=None, help="Limita batches por época para teste rápido; não usar em resultado final")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs precisa ser pelo menos 1")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps precisa ser pelo menos 1")
    configure_tensorflow()
    selected = list(EXPERIMENTS) if args.all else [args.experiment]
    for exp_id in selected:
        train_one(exp_id, args.epochs, args.force, args.max_steps)


if __name__ == "__main__":
    main()
