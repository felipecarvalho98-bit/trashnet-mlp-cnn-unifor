from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import (
    BATCH_SIZE, CLASSES, CPU_BASELINE, EARLY_STOPPING_PATIENCE, IMAGE_HEIGHT,
    IMAGE_WIDTH, LEARNING_RATE, MAX_EPOCHS, MIN_LEARNING_RATE,
    REDUCE_LR_PATIENCE, SEED, OUTPUT_DIR,
)
from .data_utils import load_or_prepare
from .gpu_monitor import GPUUsageMonitor
from .models import TrashNetCNN, count_parameters


class TrashNetImages(Dataset):
    def __init__(self, frame: pd.DataFrame, dataset_root: Path) -> None:
        self.frame = frame.reset_index(drop=True)
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.dataset_root / row.relative_path
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        # PyTorch trabalha com canais na primeira dimensão: (C, altura, largura).
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, int(row.class_index)


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def make_directml_device(device_index: int, expected_gpu: str):
    if os.name != "nt":
        raise RuntimeError("Este pacote foi preparado para Windows + DirectML. Não será usado CPU como substituto.")
    try:
        import torch_directml
    except Exception as exc:
        raise RuntimeError(
            "Não consegui importar torch_directml. Execute primeiro PREPARAR_AMBIENTE.bat "
            "com Python 3.12 de 64 bits."
        ) from exc

    try:
        count_fn = getattr(torch_directml, "device_count", None)
        count = int(count_fn()) if callable(count_fn) else None
        if count is not None and device_index >= count:
            raise RuntimeError(f"Índice DirectML {device_index} não existe. Dispositivos detectados: {count}.")
        device = torch_directml.device(device_index)
        name_fn = getattr(torch_directml, "device_name", None)
        device_name = name_fn(device_index) if callable(name_fn) else "Nome não informado pelo backend"
    except Exception as exc:
        raise RuntimeError(f"Não foi possível abrir o dispositivo DirectML {device_index}: {exc}") from exc

    if not isinstance(device, torch.device) and "privateuseone" not in str(device).lower():
        raise RuntimeError(f"O dispositivo retornado não parece ser DirectML: {device}")
    if expected_gpu and expected_gpu.casefold() not in str(device_name).casefold():
        raise RuntimeError(
            f"O dispositivo DirectML {device_index} foi identificado como '{device_name}', "
            f"mas esperávamos encontrar '{expected_gpu}'. Não vou usar outra GPU por engano. "
            "Verifique o índice DirectML antes de treinar."
        )
    return device, str(device_name), count


def smoke_test_directml(device, device_name: str, output_dir: Path) -> tuple[dict, nn.Module]:
    """Testa a CNN E3 completa, loss, backward e Adam no dispositivo DirectML."""
    set_seeds()
    model = TrashNetCNN(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    images = torch.randn((4, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32, device=device)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    losses = []

    model.train()
    for step in range(2):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().to("cpu").item())
        if not np.isfinite(value):
            raise RuntimeError("A CNN completa produziu uma perda inválida durante o teste DirectML.")
        losses.append(value)

    report = {
        "status": "APROVADO",
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "model_parameters": count_parameters(model),
        "input_shape_nchw": [4, 3, IMAGE_HEIGHT, IMAGE_WIDTH],
        "forward_backward_adam_steps": len(losses),
        "loss_values": losses,
        "pretrained_weights": False,
        "note": "Este teste verifica a arquitetura E3 completa no DirectML, não apenas uma CNN reduzida.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teste_cnn_completa_directml.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Mantém o modelo vivo até concluir a primeira leitura de memória do processo pela telemetria.
    return report, model


def move_batch(images: torch.Tensor, labels: torch.Tensor, device):
    # Não há conversão automática para CPU/CUDA: ambos os tensores vão diretamente ao DirectML.
    return images.to(device), labels.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    true_values: list[int] = []
    predicted_values: list[int] = []

    for images, labels in loader:
        images, labels = move_batch(images, labels, device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        loss_sum += float(loss.detach().to("cpu").item()) * batch_size
        predictions = logits.detach().to("cpu").numpy().argmax(axis=1)
        true_values.extend(labels.detach().to("cpu").numpy().astype(int).tolist())
        predicted_values.extend(predictions.astype(int).tolist())
        sample_count += batch_size

    return loss_sum / max(sample_count, 1), float(accuracy_score(true_values, predicted_values))


def evaluate_loader(model, loader, criterion, device) -> dict:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    true_values: list[int] = []
    predicted_values: list[int] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = move_batch(images, labels, device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1).detach().to("cpu").numpy()
            predictions = probabilities.argmax(axis=1)
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.detach().to("cpu").item()) * batch_size
            sample_count += batch_size
            true_values.extend(labels.detach().to("cpu").numpy().astype(int).tolist())
            predicted_values.extend(predictions.astype(int).tolist())
            all_probabilities.extend(probabilities.astype(float).tolist())

    return {
        "loss": loss_sum / max(sample_count, 1),
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "macro_precision": float(precision_score(true_values, predicted_values, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true_values, predicted_values, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true_values, predicted_values, labels=range(len(CLASSES)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_values, predicted_values, labels=range(len(CLASSES)), average="weighted", zero_division=0)),
        "y_true": true_values,
        "y_pred": predicted_values,
        "probabilities": all_probabilities,
    }


def save_checkpoint(model, path: Path, epoch: int, val_macro_f1: float) -> None:
    # Grava pesos no CPU para o arquivo ser portátil e não depender da sessão DirectML.
    state = {name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()}
    torch.save({"epoch": epoch, "val_macro_f1": val_macro_f1, "state_dict": state}, path)


def save_curves(history: pd.DataFrame, output_dir: Path) -> None:
    if history.empty:
        return
    specifications = [
        ("accuracy", "val_accuracy", "Acurácia por época", "Acurácia", "curva_acuracia.png"),
        ("loss", "val_loss", "Perda por época", "Loss", "curva_loss.png"),
        (None, "val_macro_f1", "F1 macro da validação", "F1 macro", "curva_f1_macro.png"),
    ]
    for train_col, val_col, title, ylabel, filename in specifications:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if train_col and train_col in history:
            ax.plot(history.epoch, history[train_col], label="Treino")
        ax.plot(history.epoch, history[val_col], label="Validação")
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)


def save_test_outputs(result: dict, test_frame: pd.DataFrame, output_dir: Path) -> dict:
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))), target_names=list(CLASSES),
        output_dict=True, zero_division=0,
    )
    class_rows = []
    for class_name in CLASSES:
        values = report[class_name]
        class_rows.append({
            "classe": class_name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })
    pd.DataFrame(class_rows).to_csv(output_dir / "metricas_por_classe_teste.csv", index=False, encoding="utf-8-sig")

    predictions = test_frame[["relative_path", "class_name", "class_index"]].copy().reset_index(drop=True)
    predictions = predictions.rename(columns={"class_name": "classe_real", "class_index": "indice_real"})
    predictions["classe_prevista"] = [CLASSES[index] for index in y_pred]
    predictions["confianca_prevista"] = [max(row) for row in result["probabilities"]]
    predictions["acertou"] = [a == b for a, b in zip(y_true, y_pred)]
    for class_index, class_name in enumerate(CLASSES):
        predictions[f"prob_{class_name}"] = [row[class_index] for row in result["probabilities"]]
    predictions.to_csv(output_dir / "previsoes_teste.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_normalized = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))), normalize="true")
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(output_dir / "matriz_confusao_teste.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_normalized, index=CLASSES, columns=CLASSES).to_csv(output_dir / "matriz_confusao_teste_normalizada.csv", encoding="utf-8-sig")

    for matrix, title, filename, fmt in [
        (cm, "Matriz de confusão — contagens", "matriz_confusao_contagens.png", "d"),
        (cm_normalized, "Matriz de confusão — proporção por classe real", "matriz_confusao_percentual.png", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(image, ax=ax)
        ax.set(
            xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
            xticklabels=CLASSES, yticklabels=CLASSES,
            ylabel="Classe real", xlabel="Classe prevista", title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
        threshold = float(np.nanmax(matrix)) / 2 if np.size(matrix) else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value_text = format(int(matrix[i, j]), fmt) if fmt == "d" else format(float(matrix[i, j]), fmt)
                ax.text(j, i, value_text, ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)

    return {
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
        "test_macro_precision": result["macro_precision"],
        "test_macro_recall": result["macro_recall"],
        "test_macro_f1": result["macro_f1"],
        "test_weighted_f1": result["weighted_f1"],
        "test_samples": len(y_true),
        "classification_report": report,
    }


def save_gpu_telemetry_plots(output_dir: Path) -> None:
    csv_path = output_dir / "gpu_monitoramento.csv"
    if not csv_path.exists():
        return
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return
    if frame.empty:
        return
    x = frame["elapsed_seconds"] if "elapsed_seconds" in frame else np.arange(len(frame))
    if "gpu_engine_max_percent" in frame and frame["gpu_engine_max_percent"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(x, frame["gpu_engine_max_percent"], label="Maior motor GPU do processo")
        ax.set_xlabel("Tempo desde o início do monitoramento (s)")
        ax.set_ylabel("Utilização amostrada (%)")
        ax.set_title("Utilização da GPU durante a execução")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "grafico_uso_gpu.png", dpi=170)
        plt.close(fig)

    memory_columns = [
        ("gpu_process_dedicated_mb", "Memória dedicada do processo"),
        ("gpu_process_shared_mb", "Memória compartilhada do processo"),
    ]
    available = [(column, label) for column, label in memory_columns if column in frame and frame[column].notna().any()]
    if available:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for column, label in available:
            ax.plot(x, frame[column], label=label)
        ax.set_xlabel("Tempo desde o início do monitoramento (s)")
        ax.set_ylabel("Memória (MB)")
        ax.set_title("Memória GPU atribuída ao processo Python")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "grafico_memoria_gpu.png", dpi=170)
        plt.close(fig)


def run_training(args) -> int:
    set_seeds()
    run_name = args.run_name or datetime.now().strftime("E3_DirectML_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"A pasta de execução já existe e não será sobrescrita: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparando o TrashNet e recriando a divisão estratificada...")
    full_frame, dataset_root, dataset_metadata = load_or_prepare()
    train_frame = full_frame[full_frame.split == "train"].reset_index(drop=True)
    val_frame = full_frame[full_frame.split == "validation"].reset_index(drop=True)
    test_frame = full_frame[full_frame.split == "test"].reset_index(drop=True)
    print(f"Imagens: {len(full_frame)} | treino: {len(train_frame)} | validação: {len(val_frame)} | teste: {len(test_frame)}")

    device, device_name, device_count = make_directml_device(args.device_index, args.gpu_name_contains)
    print(f"DirectML: {device} | GPU informada: {device_name} | dispositivos: {device_count}")
    # O monitor começa antes do teste de arquitetura para tentar capturar também a atividade do próprio teste.
    monitor = GPUUsageMonitor(output_dir, interval_seconds=1.0)
    monitor.start()
    try:
        smoke, smoke_model = smoke_test_directml(device, device_name, output_dir)
        print("Teste da CNN completa no DirectML: APROVADO")
        # Aguarda algumas amostras para confirmar que o Windows está expondo os dados do processo.
        deadline = time.perf_counter() + 12.0
        while time.perf_counter() < deadline:
            good_rows = [row for row in monitor.rows if row["counter_status"] == "OK"]
            has_activity = any(
                (row.get("gpu_engine_instances") or 0) > 0
                or (row.get("gpu_process_dedicated_mb") or 0) > 0
                or (row.get("gpu_process_shared_mb") or 0) > 0
                for row in good_rows
            )
            if has_activity:
                break
            time.sleep(0.25)
        good_rows = [row for row in monitor.rows if row["counter_status"] == "OK"]
        has_activity = any(
            (row.get("gpu_engine_instances") or 0) > 0
            or (row.get("gpu_process_dedicated_mb") or 0) > 0
            or (row.get("gpu_process_shared_mb") or 0) > 0
            for row in good_rows
        )
        if not good_rows:
            monitor.stop()
            last = monitor.rows[-1]["counter_status"] if monitor.rows else "nenhuma amostra recebida"
            raise RuntimeError(
                "A CNN completa passou no DirectML, mas o monitor do Windows não conseguiu consultar "
                f"os contadores de GPU: {last}. O treinamento longo não começou."
            )
        if not has_activity:
            monitor.stop()
            raise RuntimeError(
                "Os contadores do Windows responderam, mas não foi observada utilização ou memória "
                "GPU atribuída ao processo Python durante o teste. O treinamento longo não começou "
                "para evitar uma execução sem telemetria útil."
            )
        (output_dir / "teste_contadores_gpu.json").write_text(
            json.dumps(good_rows[-1], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Contadores do Windows: GPU ou memória do processo identificada.")
        if args.check_only:
            monitor_summary = monitor.stop()
            save_gpu_telemetry_plots(output_dir)
            (output_dir / "gpu_monitoramento_resumo.json").write_text(
                json.dumps(monitor_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"Relatórios salvos em: {output_dir}")
            return 0
        del smoke_model
    except Exception:
        monitor.stop()
        raise

    # A semente é reaplicada depois do smoke test para que ele não altere a inicialização do experimento.
    set_seeds()
    model = TrashNetCNN(num_classes=len(CLASSES)).to(device)
    parameters = count_parameters(model)
    if parameters != 304_294:
        raise RuntimeError(f"Arquitetura inesperada: {parameters:,} parâmetros; esperado 304.294.")

    train_loader = DataLoader(
        TrashNetImages(train_frame, dataset_root), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False,
        generator=torch.Generator().manual_seed(SEED), drop_last=False,
    )
    val_loader = DataLoader(
        TrashNetImages(val_frame, dataset_root), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
    )
    test_loader = DataLoader(
        TrashNetImages(test_frame, dataset_root), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False, drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )
    history_rows: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    overall_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    interrupted = False

    print("\nIniciando E3 do zero na GPU DirectML (sem pesos pré-treinados).")
    print(f"CNN: 128x96 | lote: {BATCH_SIZE} | até {args.epochs} épocas | parâmetros: {parameters:,}")
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
            validation = evaluate_loader(model, val_loader, criterion, device)
            epoch_seconds = time.perf_counter() - epoch_start
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scheduler.step(validation["loss"])
            try:
                ram_mb = process.memory_info().rss / (1024 ** 2)
            except Exception:
                ram_mb = None

            row = {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_accuracy,
                "val_loss": validation["loss"],
                "val_accuracy": validation["accuracy"],
                "val_macro_precision": validation["macro_precision"],
                "val_macro_recall": validation["macro_recall"],
                "val_macro_f1": validation["macro_f1"],
                "learning_rate": learning_rate,
                "epoch_time_seconds": epoch_seconds,
                "process_ram_mb_at_epoch_end": ram_mb,
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(output_dir / "historico_treinamento.csv", index=False, encoding="utf-8-sig")

            if validation["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(model, output_dir / "melhor_modelo.pt", epoch, best_macro_f1)
                print(f"  Melhor checkpoint atualizado (val F1 macro={best_macro_f1:.4f}).")
            else:
                epochs_without_improvement += 1

            print(
                f"Época {epoch:02d}/{args.epochs} | loss {train_loss:.4f} | acc {train_accuracy:.4f} "
                f"| val_loss {validation['loss']:.4f} | val_acc {validation['accuracy']:.4f} "
                f"| val_F1_macro {validation['macro_f1']:.4f} | {epoch_seconds:.1f}s"
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Parada antecipada: F1 macro da validação não melhorou por {EARLY_STOPPING_PATIENCE} épocas.")
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nExecução interrompida pelo usuário. O histórico e o melhor checkpoint já salvos serão mantidos.")
    except Exception:
        (output_dir / "erro_treinamento.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        monitor_summary = monitor.stop()

    save_gpu_telemetry_plots(output_dir)
    training_seconds = time.perf_counter() - overall_start
    history = pd.DataFrame(history_rows)
    save_curves(history, output_dir)
    if best_epoch is None or not (output_dir / "melhor_modelo.pt").exists():
        raise RuntimeError("Nenhuma época conseguiu salvar um checkpoint pela macro-F1 de validação.")

    checkpoint = torch.load(output_dir / "melhor_modelo.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate_loader(model, test_loader, criterion, device)
    test_summary = save_test_outputs(test_result, test_frame, output_dir)
    save_checkpoint(model, output_dir / "modelo_final_avaliado.pt", best_epoch, best_macro_f1)

    hardware = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_directml": "0.2.5.dev240914 (instalado pelo pacote requirements)",
        "device_backend": str(device),
        "device_name_reported_by_directml": device_name,
        "directml_device_count": device_count,
        "logical_cpu_count": os.cpu_count(),
        "system_ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "training_device": "DirectML GPU (sem fallback para CPU)",
    }
    metadata = {
        "experiment": "E3_PyTorch_DirectML",
        "description": "CNN 128x96, sem pesos de classe, inicialização aleatória e treinamento do zero.",
        "architecture": "3 blocos; cada bloco possui 2 Conv2D + ReLU + MaxPooling2D; GAP; Dense(128); Dropout(0.35); Dense(6)",
        "input_width_height": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "normalization": "RGB float32 dividido por 255 (0 a 1)",
        "augmentation": "Não utilizada",
        "pretrained_weights": False,
        "class_weights": False,
        "optimizer": "Adam",
        "learning_rate_initial": LEARNING_RATE,
        "loss": "CrossEntropyLoss (equivalente à entropia cruzada categórica com logits)",
        "batch_size": BATCH_SIZE,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_rows),
        "best_epoch_by_validation_macro_f1": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_on_plateau": {"monitor": "val_loss", "factor": 0.5, "patience": REDUCE_LR_PATIENCE, "min_lr": MIN_LEARNING_RATE},
        "total_training_seconds_including_evaluation_and_monitor_shutdown": round(training_seconds, 3),
        "training_epoch_seconds_sum": round(float(history.epoch_time_seconds.sum()) if not history.empty else 0, 3),
        "model_parameters": parameters,
        "model_size_mb": round((output_dir / "modelo_final_avaliado.pt").stat().st_size / (1024 ** 2), 3),
        "interrupted_by_user": interrupted,
        "seed": SEED,
        "dataset": dataset_metadata,
        "hardware": hardware,
        "gpu_monitor_summary": monitor_summary,
        "test_metrics": test_summary,
        "official_cpu_baseline": CPU_BASELINE,
        "comparison_accuracy_difference_percentage_points": round((test_summary["test_accuracy"] - CPU_BASELINE["test_accuracy"]) * 100, 3),
        "comparison_macro_f1_difference_percentage_points": round((test_summary["test_macro_f1"] - CPU_BASELINE["test_macro_f1"]) * 100, 3),
        "comparison_warning": CPU_BASELINE["note"],
    }
    (output_dir / "resumo_execucao.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "resumo_execucao.txt").open("w", encoding="utf-8") as stream:
        stream.write("EXPERIMENTO E3 — PYTORCH + DIRECTML\n")
        stream.write("Treinamento com pesos inicializados do zero; sem modelo pré-treinado.\n\n")
        stream.write(f"GPU informada: {device_name}\nÉpocas concluídas: {len(history_rows)}\nMelhor época (val macro-F1): {best_epoch}\n")
        stream.write(f"Acurácia teste: {test_summary['test_accuracy']:.4f}\nF1 macro teste: {test_summary['test_macro_f1']:.4f}\n")
        stream.write(f"Tempo total (s): {training_seconds:.2f}\nParâmetros: {parameters:,}\n")
        stream.write(f"Amostras de telemetria GPU válidas: {monitor_summary['successful_gpu_samples']}\n")
        stream.write("\nConsulte gpu_monitoramento.csv para as leituras coletadas ao longo do treinamento.\n")

    print("\nTreinamento E3 finalizado.")
    print(f"Melhor época: {best_epoch} | teste accuracy: {test_summary['test_accuracy']:.4f} | teste macro-F1: {test_summary['test_macro_f1']:.4f}")
    print(f"Resultados separados salvos em: {output_dir}")
    print("O projeto original e os resultados oficiais não foram alterados.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa a CNN E3 do TrashNet em DirectML.")
    parser.add_argument("--device-index", type=int, default=0, help="Índice do dispositivo DirectML (padrão: 0).")
    parser.add_argument("--gpu-name-contains", default="RX 5500 XT", help="Trecho obrigatório do nome da GPU (padrão: RX 5500 XT; use aspas vazias para desativar).")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Máximo de épocas (padrão: {MAX_EPOCHS}).")
    parser.add_argument("--run-name", default=None, help="Nome opcional para a pasta desta execução.")
    parser.add_argument("--check-only", action="store_true", help="Testa a CNN E3 completa no DirectML, sem iniciar o treinamento.")
    args = parser.parse_args()
    if args.device_index < 0:
        parser.error("--device-index precisa ser zero ou maior.")
    if not 1 <= args.epochs <= MAX_EPOCHS:
        parser.error(f"--epochs deve estar entre 1 e {MAX_EPOCHS}.")

    try:
        return run_training(args)
    except Exception as exc:
        print(f"\nFALHA: {type(exc).__name__}: {exc}")
        print("Nenhum fallback automático para CPU foi realizado.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
