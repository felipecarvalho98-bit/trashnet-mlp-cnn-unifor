from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from src.common import (
    CLASSES, EXPERIMENTS, FIG_DIR, HISTORY_DIR, METRICS_DIR, MODELS_DIR,
    OUTPUT_DIR, find_dataset_dir, load_json, make_output_dirs, save_json,
    validate_reference_dataset_counts, validate_splits,
)
from src.train import build_dataset, configure_tensorflow


def _predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    return model.predict(dataset, verbose=0)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _make_validation_scores(dataset_dir: Path, splits: dict, requested_batch_size: int | None) -> dict:
    scores = {}
    for experiment, cfg in EXPERIMENTS.items():
        model_path = MODELS_DIR / f"modelo_{experiment}.keras"
        config_path = METRICS_DIR / f"config_{experiment}.json"
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Faltam arquivos de {experiment}. Execute o treinamento dos seis modelos antes da avaliação."
            )
        width, height = cfg["width"], cfg["height"]
        config = load_json(config_path)
        inference_batch_size = requested_batch_size or int(config["batch_size_used"])
        val_ds = build_dataset(
            dataset_dir, splits["paths"]["val"], splits["labels"]["val"],
            width, height, batch_size=inference_batch_size, training=False,
        )
        model = tf.keras.models.load_model(model_path)
        probabilities = _predict(model, val_ds)
        predictions = probabilities.argmax(axis=1)
        scores[experiment] = float(
            f1_score(splits["labels"]["val"], predictions, average="macro", zero_division=0)
        )
        print(f"{experiment}: F1 macro de validação = {scores[experiment]:.4f}")
        del model, probabilities, predictions, val_ds
        tf.keras.backend.clear_session()
    return scores


def _create_learning_curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, experiment in zip(axes.ravel(), EXPERIMENTS):
        history = load_json(HISTORY_DIR / f"historico_{experiment}.json")
        epochs = np.arange(1, len(history.get("loss", [])) + 1)
        ax.plot(epochs, history.get("loss", []), label="Perda - treino")
        ax.plot(epochs, history.get("val_loss", []), label="Perda - validação")
        ax.set_xlabel("Época")
        ax.set_ylabel("Perda")
        ax.set_title(experiment)
        ax.grid(alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(epochs, history.get("accuracy", []), linestyle="--", label="Acurácia - treino")
        ax2.plot(epochs, history.get("val_accuracy", []), linestyle=":", label="Acurácia - validação")
        ax2.set_ylabel("Acurácia")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="best")
    fig.suptitle("Curvas de aprendizagem dos seis experimentos")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_curvas_aprendizagem.png", dpi=180)
    plt.close(fig)


def _create_confusion_figure(results: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, experiment in zip(axes.ravel(), EXPERIMENTS):
        matrix = np.asarray(results[experiment]["confusion_matrix"], dtype=int)
        row_totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0
        ) * 100.0
        image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=7)
        ax.set_xlabel("Prevista")
        ax.set_ylabel("Real")
        ax.set_title(experiment)
        for row in range(len(CLASSES)):
            for col in range(len(CLASSES)):
                ax.text(
                    col, row, f"{matrix[row, col]}\n{percentages[row, col]:.0f}%",
                    ha="center", va="center", fontsize=6,
                    color="white" if percentages[row, col] >= 60 else "black",
                )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7, label="Percentual por classe real (%)")
    fig.suptitle("Matrizes de confusão: contagem e percentual por linha")
    fig.savefig(FIG_DIR / "fig6_matrizes_confusao.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _create_resolution_figure(results: dict) -> None:
    experiments = ["E3", "E5", "E6"]
    labels = [
        f"{EXPERIMENTS[e]['width']} × {EXPERIMENTS[e]['height']}"
        for e in experiments
    ]
    accuracy = [results[e]["accuracy_test"] for e in experiments]
    macro_f1 = [results[e]["macro_f1_test"] for e in experiments]
    mean_seconds = [results[e]["training_mean_seconds_per_epoch"] for e in experiments]
    total_seconds = [results[e]["training_total_seconds"] for e in experiments]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(experiments))
    width = 0.35
    axes[0].bar(x - width / 2, accuracy, width, label="Acurácia")
    axes[0].bar(x + width / 2, macro_f1, width, label="F1 macro")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Métrica no teste")
    axes[0].set_title("Desempenho")
    axes[0].legend(fontsize=8)

    axes[1].bar(labels, mean_seconds)
    axes[1].set_ylabel("Segundos")
    axes[1].set_title("Tempo médio por época")

    axes[2].bar(labels, total_seconds)
    axes[2].set_ylabel("Segundos")
    axes[2].set_title("Tempo total de treinamento")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("CNN sem pesos de classe: efeito da resolução e custo")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_resolucao_e_custo.png", dpi=180)
    plt.close(fig)


def _create_error_figure(dataset_dir: Path, splits: dict, best_experiment: str,
                         best_predictions: np.ndarray, best_probabilities: np.ndarray) -> list[dict]:
    y_true = np.asarray(splits["labels"]["test"], dtype=int)
    errors = np.flatnonzero(best_predictions != y_true)[:6]
    fig, axes = plt.subplots(1, 6, figsize=(16, 4))
    for ax in axes.ravel():
        ax.axis("off")
    records = []
    if len(errors) == 0:
        fig.suptitle(f"{best_experiment}: nenhum erro no conjunto de teste")
    else:
        for ax, index in zip(axes.ravel(), errors):
            relative_path = splits["paths"]["test"][int(index)]
            with Image.open(dataset_dir / relative_path) as source:
                ax.imshow(ImageOps.exif_transpose(source).convert("RGB"))
            predicted_class = int(best_predictions[index])
            probability = float(best_probabilities[index, predicted_class])
            real_class = int(y_true[index])
            ax.set_title(
                f"Real: {CLASSES[real_class]}\nPrev.: {CLASSES[predicted_class]}\nP={probability:.3f}",
                fontsize=7,
            )
            ax.axis("off")
            records.append({
                "path": relative_path,
                "real_class": CLASSES[real_class],
                "predicted_class": CLASSES[predicted_class],
                "predicted_class_probability": probability,
            })
    fig.suptitle(f"Até seis erros do modelo selecionado pela validação ({best_experiment})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_erros_modelo_referencia.png", dpi=180)
    plt.close(fig)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia os seis modelos e gera tabelas e figuras.")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Lote de inferência. Por padrão, usa o lote registrado no treinamento de cada modelo.",
    )
    args = parser.parse_args()

    make_output_dirs()
    dataset_dir = find_dataset_dir()
    validate_reference_dataset_counts(dataset_dir)
    splits = load_json(OUTPUT_DIR / "splits.json")
    validate_splits(splits)
    configure_tensorflow()

    # Primeiro calcula apenas F1 macro da validação e escolhe o modelo de referência.
    validation_scores = _make_validation_scores(dataset_dir, splits, args.batch_size)
    best_experiment = max(validation_scores, key=validation_scores.get)
    print(f"\nModelo escolhido pelo maior F1 macro de validação: {best_experiment} ({validation_scores[best_experiment]:.4f})")

    results = {}
    per_class_rows = []
    performance_rows = []
    cost_rows = []
    best_probabilities = None
    best_predictions = None

    # Depois da seleção, todos os seis modelos são avaliados no mesmo conjunto de teste.
    for experiment, cfg in EXPERIMENTS.items():
        width, height = cfg["width"], cfg["height"]
        config = load_json(METRICS_DIR / f"config_{experiment}.json")
        inference_batch_size = args.batch_size or int(config["batch_size_used"])
        test_ds = build_dataset(
            dataset_dir, splits["paths"]["test"], splits["labels"]["test"],
            width, height, inference_batch_size, False,
        )
        model = tf.keras.models.load_model(MODELS_DIR / f"modelo_{experiment}.keras")
        probabilities = _predict(model, test_ds)
        predictions = probabilities.argmax(axis=1)
        y_test = np.asarray(splits["labels"]["test"], dtype=int)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, predictions, labels=range(len(CLASSES)), zero_division=0
        )
        matrix = confusion_matrix(y_test, predictions, labels=range(len(CLASSES)))
        metrics = {
            "experiment": experiment,
            "validation_macro_f1": validation_scores[experiment],
            "accuracy_test": float(accuracy_score(y_test, predictions)),
            "macro_f1_test": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "per_class": {
                class_name: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, class_name in enumerate(CLASSES)
            },
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y_test, predictions, labels=range(len(CLASSES)),
                target_names=CLASSES, zero_division=0, output_dict=True,
            ),
        }
        save_json(metrics, METRICS_DIR / f"metricas_{experiment}.json")
        results[experiment] = metrics

        config = load_json(METRICS_DIR / f"config_{experiment}.json")
        per_class_rows.extend([
            [experiment, class_name, precision[i], recall[i], f1[i], support[i]]
            for i, class_name in enumerate(CLASSES)
        ])
        performance_rows.append([
            experiment, cfg["model"].upper(), f"{width}x{height}",
            "Sim" if cfg["class_weights"] else "Não",
            validation_scores[experiment], metrics["accuracy_test"], metrics["macro_f1_test"],
        ])
        cost_rows.append([
            experiment, cfg["model"].upper(), f"{width}x{height}",
            config["batch_size_used"], config["epochs_executed"],
            config["total_parameters"], config["trainable_parameters"], config["model_size_mb"],
            config["training_total_seconds"], config["training_mean_seconds_per_epoch"],
            config["ram_peak_sampled_mb"], config["gpu_tensorflow_allocator_peak_mb"],
            config.get("gpu_process_peak_nvidia_smi_mb"),
            config.get("gpu_utilization_mean_sampled_percent"),
            config.get("gpu_utilization_max_sampled_percent"),
            "; ".join(config["hardware"]["gpu_names"]) if config["hardware"]["gpu_names"] else "CPU",
        ])

        if experiment == best_experiment:
            best_probabilities = probabilities.copy()
            best_predictions = predictions.copy()

        del model, probabilities, predictions, test_ds
        tf.keras.backend.clear_session()

    _write_csv(
        METRICS_DIR / "tabela5_custo.csv",
        ["Experimento", "Modelo", "Resolução (LxA)", "Lote", "Épocas", "Parâmetros totais",
         "Parâmetros treináveis", "Tamanho do modelo (MiB)", "Tempo total (s)", "Tempo médio/época (s)",
         "Pico RAM amostrado (MiB)", "Pico alocador GPU TensorFlow (MiB)",
         "Pico VRAM processo nvidia-smi (MiB)", "Utilização média GPU amostrada (%)",
         "Utilização máxima GPU amostrada (%)", "GPU/CPU"],
        cost_rows,
    )
    _write_csv(
        METRICS_DIR / "tabela6_desempenho.csv",
        ["Experimento", "Modelo", "Resolução (LxA)", "Pesos de classe", "F1 macro validação",
         "Acurácia teste", "F1 macro teste"],
        performance_rows,
    )
    _write_csv(
        METRICS_DIR / "tabela7_por_classe.csv",
        ["Experimento", "Classe", "Precisão", "Recall", "F1", "Suporte"],
        per_class_rows,
    )

    _create_learning_curves()
    _create_confusion_figure(results)
    _create_resolution_figure(results)
    error_records = _create_error_figure(
        dataset_dir, splits, best_experiment, best_predictions, best_probabilities
    )
    save_json({
        "selected_by_validation_macro_f1": best_experiment,
        "validation_macro_f1_by_experiment": validation_scores,
        "test_errors_shown": error_records,
        "test_evaluated_after_model_selection": True,
    }, METRICS_DIR / "modelo_referencia_erros.json")
    save_json(results, METRICS_DIR / "resultados_completos.json")

    print("\nTabelas geradas:")
    for filename in ["tabela5_custo.csv", "tabela6_desempenho.csv", "tabela7_por_classe.csv"]:
        print(f" - outputs/metricas/{filename}")
    print("Figuras geradas em outputs/figuras (figuras 5 a 8).")


if __name__ == "__main__":
    main()
