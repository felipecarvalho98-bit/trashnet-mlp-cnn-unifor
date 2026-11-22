"""Recupera a avaliação do YOLO a partir de best.pt, sem novo treinamento.

Se a pasta dataset_classification estiver incompleta, ela é reconstruída a partir
 de outputs/preparacao/splits.csv e do mesmo dataset original. Os checkpoints em
runs/yolo26n_cls nunca são apagados ou modificados por este script.
"""
from __future__ import annotations

import json
import shutil
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

from .config import CLASS_NAMES, CLASS_TO_INDEX, FINAL_DIR, IMAGE_EXTENSIONS, OUTPUT_DIR
from .data_utils import load_splits
from .train_yolo_bonus import _prepare_yolo_folders

YOLO_DIR = Path(OUTPUT_DIR) / "yolo_bonus"
RUN_DIR = YOLO_DIR / "runs" / "yolo26n_cls"
BEST_PT = RUN_DIR / "weights" / "best.pt"
LAST_PT = RUN_DIR / "weights" / "last.pt"
DATASET_DIR = YOLO_DIR / "dataset_classification"
EXPECTED_SPLITS = {"train": 1766, "val": 379, "test": 379}


def _image_count(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _class_counts_match(folder: Path, expected: pd.DataFrame) -> bool:
    """Confere quantidade de imagens por classe, não apenas se a pasta existe."""
    if not folder.is_dir():
        return False
    actual = {}
    for class_dir in folder.iterdir():
        if class_dir.is_dir():
            actual[class_dir.name.lower()] = sum(
                1 for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
    wanted = expected.groupby("class_name").size().to_dict()
    return actual == {str(k).lower(): int(v) for k, v in wanted.items()}


def ensure_dataset_splits() -> Path:
    """Reconstrói train/val/test apenas se necessário, usando a divisão original."""
    split_df, dataset_root = load_splits()
    counts = split_df["split"].value_counts().to_dict()
    if counts != {"train": 1766, "validation": 379, "test": 379}:
        raise ValueError(
            "A divisão encontrada não corresponde à utilizada nos experimentos E1–E6. "
            f"Contagens encontradas: {counts}"
        )

    expected_metadata = OUTPUT_DIR / "preparacao" / "dataset_metadata.json"
    metadata = json.loads(expected_metadata.read_text(encoding="utf-8"))
    if int(metadata.get("total_images_after_exact_deduplication", -1)) != 2524:
        raise ValueError("O metadata não confirma o dataset de 2.524 imagens usado em E1–E6.")

    mapping = {"train": "train", "validation": "val", "test": "test"}
    complete = True
    for original_name, yolo_name in mapping.items():
        expected_rows = split_df[split_df["split"] == original_name]
        expected_count = EXPECTED_SPLITS[yolo_name]
        folder = DATASET_DIR / yolo_name
        if _image_count(folder) != expected_count or not _class_counts_match(folder, expected_rows):
            complete = False
            break

    if complete:
        print("As pastas train/val/test já estão completas; serão reutilizadas.")
        return dataset_root

    print("A estrutura dataset_classification está incompleta.")
    print("Vou reconstruí-la a partir de splits.csv, conferindo o SHA-256 de cada imagem.")
    print("Os checkpoints best.pt e last.pt serão preservados.")

    rebuilt_dir = YOLO_DIR / "dataset_classification_reconstruido"
    _prepare_yolo_folders(split_df, rebuilt_dir)

    for yolo_name, expected_count in EXPECTED_SPLITS.items():
        folder = rebuilt_dir / yolo_name
        if _image_count(folder) != expected_count:
            raise RuntimeError(
                f"A reconstrução de {yolo_name} ficou incompleta: "
                f"esperadas {expected_count}, encontradas {_image_count(folder)}."
            )

    if DATASET_DIR.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_dir = YOLO_DIR / f"dataset_classification_incompleto_{stamp}"
        DATASET_DIR.rename(old_dir)
        print("Pasta incompleta preservada em:", old_dir)
    rebuilt_dir.rename(DATASET_DIR)
    print("Estrutura reconstruída com sucesso:", DATASET_DIR)
    return dataset_root


def predict_split(model: YOLO, split_name: str, class_names: list[str], device) -> pd.DataFrame:
    """Prevê cada classe separadamente, pois o Ultralytics não percorre subpastas de classe ao receber a pasta train/val/test inteira."""
    split_folder = DATASET_DIR / split_name
    if not split_folder.is_dir():
        raise FileNotFoundError(f"Pasta da partição não encontrada: {split_folder}")

    rows = []
    for true_name in class_names:
        class_folder = split_folder / true_name
        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Pasta da classe '{true_name}' não encontrada na partição '{split_name}': {class_folder}"
            )
        image_count = _image_count(class_folder)
        if image_count == 0:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada para a classe '{true_name}' em {class_folder}"
            )

        print(f"  {true_name}: {image_count} imagens", flush=True)
        for result in model.predict(
            source=str(class_folder), imgsz=224, batch=16, device=device,
            stream=True, verbose=False,
        ):
            image_path = Path(result.path)
            probabilities = result.probs.data.detach().cpu().numpy()
            index = int(np.argmax(probabilities))
            predicted_name = str(result.names[index]).lower()
            if predicted_name not in CLASS_TO_INDEX:
                raise ValueError(f"Classe retornada pelo YOLO não reconhecida: {predicted_name}")
            rows.append({
                "arquivo": str(image_path),
                "classe_real": true_name,
                "classe_prevista": predicted_name,
                "confianca_prevista": float(probabilities[index]),
            })

    if not rows:
        raise RuntimeError(f"Nenhuma previsão foi gerada para {split_folder}.")

    predictions = pd.DataFrame(rows)
    # Confere que cada imagem da partição foi avaliada uma única vez.
    expected = _image_count(split_folder)
    if len(predictions) != expected:
        raise RuntimeError(
            f"Quantidade de previsões divergente em {split_name}: "
            f"esperadas {expected}, obtidas {len(predictions)}."
        )
    predictions.to_csv(YOLO_DIR / f"previsoes_{split_name}.csv", index=False)
    print(f"  Total previsto em {split_name}: {len(predictions)} imagens", flush=True)
    return predictions


def save_metrics(predictions: pd.DataFrame, suffix: str, class_names: list[str]) -> dict:
    y_true = predictions["classe_real"].tolist()
    y_pred = predictions["classe_prevista"].tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    pd.DataFrame({
        "classe": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    }).to_csv(YOLO_DIR / f"metricas_por_classe_{suffix}.csv", index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    row_sums = matrix.sum(axis=1, keepdims=True)
    percent = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0) * 100
    for values, fmt, title, filename in [
        (matrix, "d", "Contagens", f"matriz_confusao_{suffix}_counts.png"),
        (percent, ".1f", "Percentual por classe real (%)", f"matriz_confusao_{suffix}_percent.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe real")
        ax.set_title(f"YOLO — {suffix.upper()} — {title}")
        fig.colorbar(image, ax=ax)
        threshold = float(np.max(values)) / 2 if values.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center",
                        color="white" if values[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(YOLO_DIR / filename, dpi=160)
        plt.close(fig)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)),
        "total_images": int(len(predictions)),
    }


def save_error_analysis(test_predictions: pd.DataFrame) -> None:
    errors = test_predictions[test_predictions.classe_real != test_predictions.classe_prevista].sort_values(
        "confianca_prevista", ascending=False
    ).head(6)
    errors.to_csv(YOLO_DIR / "analise_erros_top6.csv", index=False)
    if errors.empty:
        print("Nenhum erro de classificação encontrado no teste.")
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax in axes.flat:
        ax.axis("off")
    for ax, row in zip(axes.flat, errors.itertuples(index=False)):
        with Image.open(row.arquivo) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"Real: {row.classe_real} | Prevista: {row.classe_prevista}\nConfiança: {row.confianca_prevista:.3f}")
        ax.axis("off")
    fig.suptitle("YOLO — até seis erros no conjunto de teste")
    fig.tight_layout()
    fig.savefig(YOLO_DIR / "analise_erros_top6.png", dpi=160)
    plt.close(fig)


def save_available_curves() -> tuple[int | None, float | None]:
    history_file = RUN_DIR / "results.csv"
    if not history_file.is_file():
        return None, None
    history = pd.read_csv(history_file)
    history.columns = [c.strip() for c in history.columns]
    history.to_csv(YOLO_DIR / "historico_treinamento.csv", index=False)
    epochs = np.arange(1, len(history) + 1)
    train_loss = next((c for c in history.columns if c.lower().startswith("train/") and "loss" in c.lower()), None)
    val_loss = next((c for c in history.columns if c.lower().startswith("val/") and "loss" in c.lower()), None)
    val_acc = next((c for c in history.columns if "accuracy_top1" in c.lower()), None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if train_loss:
        axes[0].plot(epochs, history[train_loss], label="Perda de treino")
    if val_loss:
        axes[0].plot(epochs, history[val_loss], label="Perda de validação")
    axes[0].set_title("Perda por época")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    if val_acc:
        axes[1].plot(epochs, history[val_acc], label="Acurácia de validação")
    axes[1].set_title("Acurácia de validação por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(YOLO_DIR / "curvas_historico_disponivel.png", dpi=160)
    plt.close(fig)
    duration = float(history["time"].max()) if "time" in history.columns else None
    return int(len(history)), duration


def main() -> None:
    if not BEST_PT.is_file():
        raise FileNotFoundError(f"Checkpoint best.pt não encontrado: {BEST_PT}")
    if not (RUN_DIR / "results.csv").is_file():
        raise FileNotFoundError(f"Histórico do treinamento não encontrado: {RUN_DIR / 'results.csv'}")

    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    dataset_root = ensure_dataset_splits()
    model = YOLO(str(BEST_PT))
    if isinstance(model.names, dict):
        class_names = [str(model.names[i]).lower() for i in sorted(model.names)]
    else:
        class_names = [str(name).lower() for name in model.names]
    expected_classes = {name.lower() for name in CLASS_NAMES}
    if set(class_names) != expected_classes:
        raise ValueError(f"Classes do checkpoint ({class_names}) diferentes das seis classes esperadas.")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Checkpoint preservado:", BEST_PT)
    print("Dataset original:", dataset_root)
    print("Classes do checkpoint:", class_names)
    print("Dispositivo de avaliação:", device)
    print("A avaliação pode levar alguns minutos em CPU; nenhum treinamento será iniciado.")

    summaries = {}
    for split_name, suffix in [("train", "treino"), ("val", "validacao"), ("test", "teste")]:
        print(f"Avaliando partição {split_name} ...")
        predictions = predict_split(model, split_name, class_names, device)
        summaries[suffix] = save_metrics(predictions, suffix, class_names)
        print(summaries[suffix])

    test_predictions = pd.read_csv(YOLO_DIR / "previsoes_test.csv")
    save_error_analysis(test_predictions)
    epochs, training_seconds = save_available_curves()

    parameter_count = sum(p.numel() for p in model.model.parameters())
    summary = {
        "status": "avaliação recuperada do checkpoint existente; sem novo treinamento",
        "checkpoint_utilizado": str(BEST_PT),
        "checkpoint_ultima_epoca": str(LAST_PT) if LAST_PT.is_file() else None,
        "epocas_registradas_no_historico": epochs,
        "tempo_total_registrado_segundos": training_seconds,
        "tempo_medio_epoca_registrado_segundos": (training_seconds / epochs) if training_seconds and epochs else None,
        "dispositivo_avaliacao": str(device),
        "dataset_original": str(dataset_root),
        "classes": class_names,
        "metricas_treino_checkpoint_final": summaries.get("treino"),
        "metricas_validacao": summaries.get("validacao"),
        "metricas_teste": summaries.get("teste"),
        "parametros": int(parameter_count),
        "tamanho_best_mb": BEST_PT.stat().st_size / (1024 ** 2),
        "observacao_curvas": (
            "O results.csv contém perda de treino e validação e acurácia de validação. "
            "A acurácia de treino por época não foi persistida na execução interrompida; "
            "não foi inventada nesta recuperação."
        ),
    }

    baseline_file = FINAL_DIR / "resumo_experimentos.csv"
    if baseline_file.is_file():
        baseline_table = pd.read_csv(baseline_file)
        e3_rows = baseline_table[baseline_table["experimento"].astype(str).str.upper() == "E3"]
        if not e3_rows.empty:
            e3 = e3_rows.iloc[0]
            comparison = pd.DataFrame([
                {
                    "modelo": "CNN E3",
                    "accuracy_teste": float(e3["accuracy_teste"]),
                    "macro_f1_teste": float(e3["macro_f1_teste"]),
                    "tempo_treinamento_segundos": float(e3["tempo_treinamento_segundos"]),
                    "tempo_medio_epoca_segundos": float(e3["tempo_medio_epoca_segundos"]),
                    "dispositivo": str(e3["dispositivo"]),
                },
                {
                    "modelo": "YOLO26n-cls (best.pt recuperado)",
                    "accuracy_teste": summaries["teste"]["accuracy"],
                    "macro_f1_teste": summaries["teste"]["macro_f1"],
                    "tempo_treinamento_segundos": training_seconds,
                    "tempo_medio_epoca_segundos": training_seconds / epochs if training_seconds and epochs else None,
                    "dispositivo": "registrado no histórico original; avaliação em " + str(device),
                },
            ])
            comparison.to_csv(YOLO_DIR / "comparacao_yolo_E3.csv", index=False)

    summary_path = YOLO_DIR / "resumo_yolo.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nAvaliação concluída. Nenhum novo treinamento foi realizado.")
    print("Resultados:", YOLO_DIR)
    print("Resumo:", summary_path)


if __name__ == "__main__":
    main()
