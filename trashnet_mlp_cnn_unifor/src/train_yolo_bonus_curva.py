"""Treina uma nova execução isolada do YOLO e grava a acurácia de treino a cada época.

Esta versão NÃO modifica outputs/yolo_bonus, onde estão os checkpoints e resultados
já obtidos. A nova execução é salva em outputs/yolo_bonus_curva_<data_hora>.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from ultralytics import YOLO

from .config import CLASS_NAMES, CLASS_TO_INDEX, FINAL_DIR, OUTPUT_DIR
from .data_utils import load_splits
from .train_yolo_bonus import _MemorySampler, _prepare_yolo_folders

EXPECTED_SPLITS = {"train": 1766, "validation": 379, "test": 379}
YOLO_SPLITS = {"train": "train", "validation": "val", "test": "test"}


def _count_images(folder: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _training_accuracy_callback(csv_path: Path, error_path: Path):
    """Calcula e grava a acurácia de treino após cada época, sem esperar o fim do fit."""
    history: list[dict] = []
    recorded_epochs: set[int] = set()

    def callback(trainer):
        epoch_number = int(trainer.epoch) + 1
        if epoch_number in recorded_epochs:
            return

        network = trainer.model
        previous_mode = network.training
        correct = 0
        total = 0

        try:
            network.eval()
            train_loader = getattr(trainer, "train_loader", None)
            if train_loader is None:
                raise RuntimeError("O treinador não disponibilizou train_loader no callback.")

            for raw_batch in train_loader:
                batch = trainer.preprocess_batch(raw_batch)
                images = batch["img"]
                targets = batch["cls"].reshape(-1).long()

                with torch.inference_mode():
                    outputs = network(images)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predicted = outputs.argmax(dim=1)

                correct += int((predicted == targets).sum().item())
                total += int(targets.numel())

            if total == 0:
                raise RuntimeError("Nenhuma imagem foi contabilizada no cálculo da acurácia de treino.")

            row = {
                "epoca": epoch_number,
                "acuracia_treino": correct / total,
                "imagens_avaliadas": total,
            }
            history.append(row)
            history.sort(key=lambda item: item["epoca"])
            pd.DataFrame(history).to_csv(csv_path, index=False, encoding="utf-8-sig")
            recorded_epochs.add(epoch_number)

            print(
                f"[Métrica extra] Época {epoch_number}: "
                f"acurácia de treino = {row['acuracia_treino']:.4f} "
                f"({total} imagens). CSV atualizado.",
                flush=True,
            )

        except Exception as exc:
            error_path.write_text(
                f"Falha ao calcular acurácia de treino na época {epoch_number}: "
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"Não foi possível registrar a acurácia de treino na época {epoch_number}. "
                f"Detalhes em {error_path}. O restante do projeto original foi preservado."
            ) from exc
        finally:
            network.train(previous_mode)

    return callback, history


def _predict_partition(model: YOLO, partition_dir: Path, device: str | int, imgsz: int) -> pd.DataFrame:
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = partition_dir / class_name
        count = _count_images(class_dir)
        if count == 0:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em {class_dir}")

        print(f"  {class_name}: {count} imagens", flush=True)
        for result in model.predict(
            source=str(class_dir), imgsz=imgsz, batch=16, device=device,
            stream=True, verbose=False,
        ):
            image_path = Path(result.path)
            probabilities = result.probs.data.detach().cpu().numpy()
            predicted_index = int(np.argmax(probabilities))
            predicted_name = str(result.names[predicted_index]).lower()
            if predicted_name not in CLASS_TO_INDEX:
                raise ValueError(f"Classe prevista não reconhecida: {predicted_name}")
            rows.append({
                "arquivo": str(image_path),
                "classe_real": class_name,
                "classe_prevista": predicted_name,
                "confianca_prevista": float(probabilities[predicted_index]),
            })

    predictions = pd.DataFrame(rows)
    expected_total = _count_images(partition_dir)
    if len(predictions) != expected_total:
        raise RuntimeError(
            f"Número de previsões divergente em {partition_dir.name}: "
            f"esperadas {expected_total}, obtidas {len(predictions)}."
        )
    return predictions


def _save_metrics(predictions: pd.DataFrame, split_name: str, output_dir: Path) -> dict:
    y_true = predictions["classe_real"].tolist()
    y_pred = predictions["classe_prevista"].tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_NAMES), zero_division=0
    )

    pd.DataFrame({
        "classe": CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    }).to_csv(output_dir / f"metricas_por_classe_{split_name}.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / f"previsoes_{split_name}.csv", index=False, encoding="utf-8-sig")

    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
    ) * 100

    for values, fmt, title, suffix in [
        (matrix, "d", "Contagens", "counts"),
        (percentages, ".1f", "Percentual por classe real (%)", "percent"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=35, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(f"YOLO — {split_name.upper()} — {title}")
        fig.colorbar(image, ax=ax)
        threshold = float(np.max(values)) / 2 if values.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(
                    j, i, format(values[i, j], fmt), ha="center", va="center",
                    color="white" if values[i, j] > threshold else "black",
                )
        fig.tight_layout()
        fig.savefig(output_dir / f"matriz_confusao_{split_name}_{suffix}.png", dpi=160)
        plt.close(fig)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0)),
        "total_images": int(len(predictions)),
    }


def _save_error_analysis(test_predictions: pd.DataFrame, output_dir: Path) -> None:
    errors = test_predictions[
        test_predictions.classe_real != test_predictions.classe_prevista
    ].sort_values("confianca_prevista", ascending=True).head(6)
    errors.to_csv(output_dir / "analise_erros_top6.csv", index=False, encoding="utf-8-sig")
    if errors.empty:
        print("Nenhum erro foi encontrado no conjunto de teste.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.itertuples(index=False)):
        with Image.open(row.arquivo) as image_file:
            ax.imshow(image_file.convert("RGB"))
        ax.set_title(
            f"Real: {row.classe_real} | Prevista: {row.classe_prevista}\n"
            f"Confiança: {row.confianca_prevista:.3f}"
        )
        ax.axis("off")
    fig.suptitle("YOLO — até seis erros no conjunto de teste")
    fig.tight_layout()
    fig.savefig(output_dir / "analise_erros_top6.png", dpi=160)
    plt.close(fig)


def _save_curves(run_dir: Path, output_dir: Path, accuracy_csv: Path) -> int:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Histórico do Ultralytics não encontrado: {results_csv}")

    history = pd.read_csv(results_csv)
    history.columns = [column.strip() for column in history.columns]
    history.to_csv(output_dir / "historico_treinamento.csv", index=False, encoding="utf-8-sig")

    train_accuracy = pd.read_csv(accuracy_csv)
    train_accuracy = train_accuracy.drop_duplicates("epoca", keep="last").sort_values("epoca")
    train_accuracy.to_csv(output_dir / "acuracia_treino_por_epoca.csv", index=False, encoding="utf-8-sig")

    train_loss = next((c for c in history.columns if c.lower().startswith("train/") and "loss" in c.lower()), None)
    val_loss = next((c for c in history.columns if c.lower().startswith("val/") and "loss" in c.lower()), None)
    val_acc = next((c for c in history.columns if "accuracy_top1" in c.lower()), None)
    if not train_loss or not val_loss or not val_acc:
        raise RuntimeError(f"Colunas necessárias não encontradas em {results_csv}: {list(history.columns)}")

    epochs = len(history)
    # Keep exactly the epochs present in Ultralytics results.csv. Some callback versions
    # can emit one final duplicate/off-by-one sample after the last completed epoch.
    train_accuracy = train_accuracy.copy()
    train_accuracy["epoca"] = train_accuracy["epoca"].astype(int)
    train_accuracy = train_accuracy[train_accuracy["epoca"].between(1, epochs)]
    train_accuracy = train_accuracy.drop_duplicates(subset=["epoca"], keep="last").sort_values("epoca")
    train_accuracy.to_csv(accuracy_csv, index=False, encoding="utf-8-sig")
    recorded = set(train_accuracy["epoca"].astype(int).tolist())
    missing = sorted(set(range(1, epochs + 1)) - recorded)
    if missing:
        raise RuntimeError(
            f"Acurácia de treino não registrada para as épocas: {missing}. "
            f"O CSV parcial foi preservado em {accuracy_csv}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epoch_numbers = np.arange(1, epochs + 1)
    axes[0].plot(epoch_numbers, history[train_loss], label="Treino")
    axes[0].plot(epoch_numbers, history[val_loss], label="Validação")
    axes[0].set_title("Perda por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Perda")
    axes[0].legend()

    axes[1].plot(train_accuracy["epoca"], train_accuracy["acuracia_treino"], label="Acurácia de treino")
    axes[1].plot(epoch_numbers, history[val_acc], label="Acurácia de validação")
    axes[1].set_title("Acurácia top-1 por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "curvas_aprendizado.png", dpi=160)
    plt.close(fig)
    return epochs


def _save_comparison_with_e3(test: dict, summary: dict, output_dir: Path) -> None:
    baseline_file = FINAL_DIR / "resumo_experimentos.csv"
    if not baseline_file.is_file():
        print(f"Aviso: resumo da CNN E3 não encontrado em {baseline_file}; comparação não foi criada.")
        return

    baseline_table = pd.read_csv(baseline_file)
    baseline_rows = baseline_table[baseline_table["experimento"].astype(str).str.upper() == "E3"]
    if baseline_rows.empty:
        print("Aviso: E3 não localizado no resumo; comparação não foi criada.")
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
    comparison.to_csv(output_dir / "comparacao_yolo_E3.csv", index=False, encoding="utf-8-sig")

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
    parser = argparse.ArgumentParser(
        description="Nova execução isolada do YOLO com registro persistente da acurácia de treino."
    )
    parser.add_argument("--model", default="yolo26n-cls.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cpu", help="Mantém CPU para comparar com a execução anterior.")
    args = parser.parse_args()

    split_file = OUTPUT_DIR / "preparacao" / "splits.csv"
    metadata_file = OUTPUT_DIR / "preparacao" / "dataset_metadata.json"
    if not split_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Não encontrei splits.csv e dataset_metadata.json em outputs/preparacao.")

    split_df, dataset_root = load_splits()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if int(metadata.get("total_images_after_exact_deduplication", -1)) != 2524:
        raise ValueError("O dataset não corresponde às 2.524 imagens usadas nos experimentos E1–E6.")
    actual_counts = split_df["split"].value_counts().to_dict()
    if actual_counts != EXPECTED_SPLITS:
        raise ValueError(f"Divisão diferente da original. Encontrado: {actual_counts}; esperado: {EXPECTED_SPLITS}.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"yolo_bonus_curva_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"A pasta de saída já existe; nada será sobrescrito: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Nova execução isolada do YOLO.", flush=True)
    print("Resultados anteriores serão preservados em outputs/yolo_bonus.", flush=True)
    print(f"Nova pasta de resultados: {output_dir}", flush=True)
    print(f"Dataset original: {dataset_root}", flush=True)

    dataset_dir = output_dir / "dataset_classification"
    print("Preparando cópia das mesmas imagens e da divisão original...", flush=True)
    _prepare_yolo_folders(split_df, dataset_dir)

    for source_split, yolo_split in YOLO_SPLITS.items():
        count = _count_images(dataset_dir / yolo_split)
        if count != EXPECTED_SPLITS[source_split]:
            raise RuntimeError(
                f"Partição {source_split}: esperadas {EXPECTED_SPLITS[source_split]}, encontradas {count}."
            )
        print(f"{source_split}: {count} imagens conferidas.", flush=True)

    accuracy_csv = output_dir / "acuracia_treino_por_epoca.csv"
    callback_error = output_dir / "erro_callback_acuracia.txt"
    train_accuracy_callback, _history = _training_accuracy_callback(accuracy_csv, callback_error)

    model = YOLO(args.model)
    run_name = "yolo26n_cls_com_curva"
    run_dir = output_dir / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"A pasta da execução já existe; nada será apagado: {run_dir}")

    model.add_callback("on_fit_epoch_end", train_accuracy_callback)
    start_time = time.perf_counter()
    with _MemorySampler() as memory_sampler:
        model.train(
            data=str(dataset_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            seed=42,
            deterministic=True,
            workers=2,
            device=args.device,
            project=str(output_dir / "runs"),
            name=run_name,
            exist_ok=False,
        )
    training_seconds = time.perf_counter() - start_time

    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if not best_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(f"Os checkpoints da nova execução não foram encontrados em {run_dir / 'weights'}.")

    epochs_completed = _save_curves(run_dir, output_dir, accuracy_csv)
    trained_model = YOLO(str(best_path))
    device = args.device

    validation_predictions = _predict_partition(trained_model, dataset_dir / "val", device, args.imgsz)
    validation_metrics = _save_metrics(validation_predictions, "validacao", output_dir)
    test_predictions = _predict_partition(trained_model, dataset_dir / "test", device, args.imgsz)
    test_metrics = _save_metrics(test_predictions, "teste", output_dir)
    _save_error_analysis(test_predictions, output_dir)

    parameter_count = sum(parameter.numel() for parameter in trained_model.model.parameters())
    summary = {
        "status": "nova execução com acurácia de treino registrada por época",
        "model_pretrained": args.model,
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "dataset_original": str(dataset_root),
        "split": EXPECTED_SPLITS,
        "class_order": list(CLASS_NAMES),
        "image_size": args.imgsz,
        "epochs_requested": args.epochs,
        "epochs_completed": epochs_completed,
        "patience": args.patience,
        "batch": args.batch,
        "device": device,
        "training_seconds": training_seconds,
        "average_epoch_seconds": training_seconds / epochs_completed if epochs_completed else None,
        "ram_peak_sampled_mb": memory_sampler.peak_mb,
        "parameters": int(parameter_count),
        "model_size_mb": best_path.stat().st_size / (1024 ** 2),
        "training_accuracy_csv": str(output_dir / "acuracia_treino_por_epoca.csv"),
        "training_curves_png": str(output_dir / "curvas_aprendizado.png"),
        "validation": validation_metrics,
        "test": test_metrics,
        "selection_note": "O melhor checkpoint é selecionado pelo processo de validação do Ultralytics; o teste é consultado depois do treinamento.",
        "training_accuracy_note": "A acurácia é calculada ao final de cada época no train_loader, com o modelo em modo avaliação, e gravada imediatamente no CSV.",
    }
    (output_dir / "resumo_yolo_curva.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_comparison_with_e3(test_metrics, summary, output_dir)

    print("\nTREINAMENTO E AVALIAÇÃO CONCLUÍDOS.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nPasta de resultados: {output_dir}", flush=True)
    print(f"Curva de acurácia: {output_dir / 'curvas_aprendizado.png'}", flush=True)
    print(f"CSV por época: {accuracy_csv}", flush=True)


if __name__ == "__main__":
    main()
