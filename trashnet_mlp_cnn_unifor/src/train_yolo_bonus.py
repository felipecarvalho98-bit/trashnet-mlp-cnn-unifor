"""Extensão opcional: classificação TrashNet com YOLO pré-treinado.

O script reutiliza outputs/preparacao/splits.csv, cria a estrutura train/val/test
exigida pelo Ultralytics e registra métricas, curvas, matriz de confusão e erros.
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from ultralytics import YOLO

from .config import CLASS_NAMES, CLASS_TO_INDEX, FINAL_DIR, OUTPUT_DIR
from .data_utils import load_splits, sha256_file


class _MemorySampler:
    """Amostra o RSS do processo durante o treinamento."""
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, process.memory_info().rss / (1024 ** 2))
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)


def _prepare_yolo_folders(split_df: pd.DataFrame, target: Path) -> None:
    """Monta train/val/test por classe a partir da divisão registrada no projeto."""
    if target.exists():
        shutil.rmtree(target)
    split_names = {"train": "train", "validation": "val", "test": "test"}
    for row in split_df.itertuples(index=False):
        split_name = split_names[row.split]
        destination_dir = target / split_name / row.class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        source = Path(row.path)
        if sha256_file(source) != row.sha256:
            raise ValueError(
                f"A imagem {row.relative_path} não corresponde ao SHA-256 registrado na divisão original. "
                "Use o mesmo ZIP do dataset da execução E1–E6."
            )
        # SHA inicial reduz o risco de nomes repetidos dentro da classe.
        suffix = source.suffix.lower()
        filename = f"{row.sha256[:16]}{suffix}"
        destination = destination_dir / filename
        if not destination.exists():
            shutil.copy2(source, destination)


def _predict_split(model: YOLO, split_dir: Path, imgsz: int) -> pd.DataFrame:
    """Gera previsões em lote e mantém o rótulo real obtido pela pasta da classe."""
    rows = []
    for result in model.predict(
        source=str(split_dir), imgsz=imgsz, batch=32, stream=True, verbose=False
    ):
        image_path = Path(result.path)
        true_name = image_path.parent.name.lower()
        if true_name not in CLASS_TO_INDEX:
            continue
        probabilities = result.probs.data.detach().cpu().numpy()
        predicted_index = int(np.argmax(probabilities))
        predicted_name = str(result.names[predicted_index]).lower()
        if predicted_name not in CLASS_TO_INDEX:
            raise ValueError(f"Classe retornada pelo YOLO não reconhecida: {predicted_name}")
        rows.append({
            "arquivo": str(image_path),
            "classe_real": true_name,
            "classe_prevista": predicted_name,
            "confianca_prevista": float(probabilities[predicted_index]),
        })
    if not rows:
        raise RuntimeError(f"Nenhuma previsão foi gerada para {split_dir}")
    return pd.DataFrame(rows)


def _save_evaluation(model: YOLO, split_dir: Path, output_dir: Path, suffix: str, imgsz: int) -> dict:
    """Calcula métricas detalhadas e salva a matriz de confusão do split."""
    predictions = _predict_split(model, split_dir, imgsz)
    y_true = predictions["classe_real"].tolist()
    y_pred = predictions["classe_prevista"].tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_NAMES), zero_division=0
    )
    class_metrics = pd.DataFrame({
        "classe": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    })
    class_metrics.to_csv(output_dir / f"metricas_por_classe_{suffix}.csv", index=False)
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100

    for values, fmt, title, filename in [
        (matrix, "d", "Contagens", f"matriz_confusao_{suffix}_counts.png"),
        (percentages, ".1f", "Percentual por classe real (%)", f"matriz_confusao_{suffix}_percent.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(f"YOLO — {suffix.upper()} — {title}")
        fig.colorbar(image, ax=ax)
        threshold = float(np.max(values)) / 2 if values.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                label = format(values[i, j], fmt)
                ax.text(j, i, label, ha="center", va="center", color="white" if values[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    predictions.to_csv(output_dir / f"previsoes_{suffix}.csv", index=False)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0)),
        "total_images": int(len(predictions)),
    }
    if suffix == "teste":
        errors = predictions[predictions.classe_real != predictions.classe_prevista].sort_values("confianca_prevista").head(6)
        errors.to_csv(output_dir / "analise_erros_top6.csv", index=False)
        if not errors.empty:
            from PIL import Image
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            for ax in axes.flat:
                ax.axis("off")
            for ax, row in zip(axes.flat, errors.itertuples(index=False)):
                with Image.open(row.arquivo) as image_file:
                    ax.imshow(image_file.convert("RGB"))
                ax.set_title(f"Real: {row.classe_real} | Prevista: {row.classe_prevista}\nConfiança: {row.confianca_prevista:.3f}")
                ax.axis("off")
            fig.suptitle("YOLO — até seis erros com menor confiança")
            fig.tight_layout()
            fig.savefig(output_dir / "analise_erros_top6.png", dpi=160)
            plt.close(fig)
    return result


def _train_accuracy_callback(history: list[dict]):
    """Registra a acurácia no conjunto de treino ao final de cada época."""
    def callback(trainer):
        network = trainer.model
        was_training = network.training
        correct = 0
        total = 0
        network.eval()
        try:
            for raw_batch in trainer.train_loader:
                batch = trainer.preprocess_batch(raw_batch)
                images = batch["img"]
                targets = batch["cls"].view(-1).long()
                with torch.inference_mode():
                    logits = network(images)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]
                    predictions = logits.argmax(dim=1)
                correct += int((predictions == targets).sum().item())
                total += int(targets.numel())
        finally:
            network.train(was_training)
        history.append({
            "epoca": int(trainer.epoch) + 1,
            "acuracia_treino": correct / total if total else 0.0,
            "imagens_avaliadas": total,
        })
    return callback


def _plot_training_history(run_dir: Path, output_dir: Path, train_accuracy: list[dict]) -> int:
    """Plota perdas e acurácias de treino e validação por época."""
    history_path = run_dir / "results.csv"
    if not history_path.exists():
        return 0
    history = pd.read_csv(history_path)
    history.columns = [column.strip() for column in history.columns]
    history.to_csv(output_dir / "historico_treinamento.csv", index=False)
    epochs = len(history)
    train_loss = next((c for c in history.columns if c.lower().startswith("train/") and "loss" in c.lower()), None)
    val_loss = next((c for c in history.columns if c.lower().startswith("val/") and "loss" in c.lower()), None)
    val_acc = next((c for c in history.columns if "accuracy_top1" in c.lower()), None)

    if train_accuracy:
        train_frame = pd.DataFrame(train_accuracy).drop_duplicates("epoca", keep="last")
        train_frame.to_csv(output_dir / "acuracia_treino_por_epoca.csv", index=False)
    else:
        train_frame = pd.DataFrame(columns=["epoca", "acuracia_treino"])

    epoch_numbers = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if train_loss:
        axes[0].plot(epoch_numbers, history[train_loss], label="Treino")
    if val_loss:
        axes[0].plot(epoch_numbers, history[val_loss], label="Validação")
    axes[0].set_title("Perda por época")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    if not train_frame.empty:
        axes[1].plot(train_frame["epoca"], train_frame["acuracia_treino"], label="Acurácia de treino")
    if val_acc:
        axes[1].plot(epoch_numbers, history[val_acc], label="Acurácia de validação")
    axes[1].set_title("Acurácia top-1 por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "curvas_aprendizado.png", dpi=160)
    plt.close(fig)
    return epochs


def _save_comparison_with_e3(test: dict, summary: dict, output_dir: Path) -> None:
    """Compara as métricas de teste e o custo do YOLO com a CNN E3."""
    baseline_file = FINAL_DIR / "resumo_experimentos.csv"
    if not baseline_file.is_file():
        return
    baseline_table = pd.read_csv(baseline_file)
    baseline_rows = baseline_table[baseline_table["experimento"].astype(str).str.upper() == "E3"]
    if baseline_rows.empty:
        return
    baseline = baseline_rows.iloc[0]
    comparison = pd.DataFrame([
        {
            "modelo": "CNN E3",
            "accuracy_teste": float(baseline["accuracy_teste"]),
            "macro_f1_teste": float(baseline["macro_f1_teste"]),
            "parametros": int(baseline["parametros"]),
            "tamanho_modelo_mb": float(baseline["tamanho_modelo_mb"]),
            "epocas": int(baseline["epocas_executadas"]),
            "tempo_treinamento_segundos": float(baseline["tempo_treinamento_segundos"]),
            "tempo_medio_epoca_segundos": float(baseline["tempo_medio_epoca_segundos"]),
            "dispositivo": str(baseline["dispositivo"]),
        },
        {
            "modelo": f"YOLO {summary['model_pretrained']}",
            "accuracy_teste": float(test["accuracy"]),
            "macro_f1_teste": float(test["macro_f1"]),
            "parametros": int(summary["parameters"]),
            "tamanho_modelo_mb": float(summary["model_size_mb"]),
            "epocas": int(summary["epochs_completed"]),
            "tempo_treinamento_segundos": float(summary["training_seconds"]),
            "tempo_medio_epoca_segundos": float(summary["average_epoch_seconds"] or 0),
            "dispositivo": str(summary["device"]),
        },
    ])
    comparison.to_csv(output_dir / "comparacao_yolo_E3.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    comparison.plot(x="modelo", y="accuracy_teste", kind="bar", legend=False, ax=axes[0], ylim=(0, 1), rot=15)
    axes[0].set_title("Acurácia no teste")
    axes[0].set_ylabel("Acurácia")
    comparison.plot(x="modelo", y="macro_f1_teste", kind="bar", legend=False, ax=axes[1], ylim=(0, 1), rot=15)
    axes[1].set_title("F1 macro no teste")
    axes[1].set_ylabel("F1 macro")
    fig.tight_layout()
    fig.savefig(output_dir / "comparacao_yolo_E3.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Objetivo extra: YOLO para classificação TrashNet")
    parser.add_argument("--model", default="yolo26n-cls.pt", help="Modelo pré-treinado de classificação")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="Ex.: 0 para GPU ou cpu")
    args = parser.parse_args()

    split_file = OUTPUT_DIR / "preparacao" / "splits.csv"
    metadata_file = OUTPUT_DIR / "preparacao" / "dataset_metadata.json"
    if not split_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            "A divisão original não está disponível. Execute 'python -m src.prepare_data' "
            "com o dataset usado nos experimentos obrigatórios antes do YOLO."
        )

    split_df, dataset_root = load_splits()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if int(metadata.get("total_images_after_exact_deduplication", -1)) != 2524:
        raise ValueError("O conjunto não corresponde aos 2.524 arquivos da execução CNN publicada; confira o dataset original.")
    expected = {"train": 1766, "validation": 379, "test": 379}
    actual = split_df["split"].value_counts().to_dict()
    if actual != expected:
        raise ValueError(f"A divisão atual {actual} difere da divisão da execução publicada {expected}.")

    output_dir = OUTPUT_DIR / "yolo_bonus"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset_classification"
    _prepare_yolo_folders(split_df, dataset_dir)

    model = YOLO(args.model)
    device = args.device or (0 if torch.cuda.is_available() else "cpu")
    run_name = Path(args.model).stem.replace("-", "_")
    run_dir = output_dir / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_accuracy_history: list[dict] = []
    model.add_callback("on_train_epoch_end", _train_accuracy_callback(train_accuracy_history))
    start = time.perf_counter()
    with _MemorySampler() as sampler:
        model.train(
            data=str(dataset_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=10,
            seed=42,
            deterministic=True,
            workers=2,
            device=device,
            project=str(output_dir / "runs"),
            name=run_name,
            exist_ok=True,
        )
    training_seconds = time.perf_counter() - start
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint final não encontrado: {best_path}")

    trained_model = YOLO(str(best_path))
    # A validação é usada para registrar o desempenho e a configuração escolhida.
    validation = _save_evaluation(trained_model, dataset_dir / "val", output_dir, "validacao", args.imgsz)
    # O conjunto de teste só é consultado depois da avaliação/seleção na validação.
    test = _save_evaluation(trained_model, dataset_dir / "test", output_dir, "teste", args.imgsz)
    epochs_done = _plot_training_history(run_dir, output_dir, train_accuracy_history)

    parameter_count = sum(parameter.numel() for parameter in trained_model.model.parameters())
    gpu_peak_mb = None
    if torch.cuda.is_available():
        gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    summary = {
        "task": "classify",
        "model_pretrained": args.model,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "checkpoint": str(best_path),
        "dataset": str(dataset_root),
        "split": expected,
        "class_order": list(CLASS_NAMES),
        "image_size": args.imgsz,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_done,
        "batch": args.batch,
        "device": str(device),
        "training_seconds": training_seconds,
        "average_epoch_seconds": training_seconds / epochs_done if epochs_done else None,
        "ram_peak_sampled_mb": sampler.peak_mb,
        "gpu_peak_allocated_mb": gpu_peak_mb,
        "parameters": int(parameter_count),
        "model_size_mb": best_path.stat().st_size / (1024 ** 2),
        "validation": validation,
        "test": test,
        "selection_note": "O desempenho de validação é registrado antes de consultar o conjunto de teste.",
    }
    (output_dir / "resumo_yolo.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_comparison_with_e3(test, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artefatos YOLO salvos em: {output_dir}")


if __name__ == "__main__":
    main()
