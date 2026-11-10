from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from src.common import (
    CLASSES, DATA_DIR, EXPECTED_CLASS_COUNTS, EXPECTED_IMAGE_SIZE, FIG_DIR,
    LABELS_PT, METRICS_DIR, OUTPUT_DIR, SEED, find_dataset_dir,
    make_output_dirs, save_json, validate_reference_dataset_counts,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SIZE = EXPECTED_IMAGE_SIZE  # largura x altura, conforme a proposta.


def _pixel_hash(image: Image.Image) -> str:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(rgb.size).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def inspect_dataset(dataset_dir: Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    corrupted: list[str] = []
    size_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)

    for class_index, class_name in enumerate(CLASSES):
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Pasta da classe ausente: {class_dir}")
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"Nenhuma imagem encontrada na classe {class_name}.")

        for image_path in files:
            relative_path = image_path.relative_to(dataset_dir).as_posix()
            try:
                # verify() verifica o arquivo inteiro; a imagem é reaberta depois para metadados.
                with Image.open(image_path) as check:
                    check.verify()
                with Image.open(image_path) as opened:
                    width_height = opened.size
                    original_mode = opened.mode
                    content_hash = _pixel_hash(opened)
                size_counts[f"{width_height[0]}x{width_height[1]}"] += 1
                mode_counts[original_mode] += 1
                hashes[content_hash].append(relative_path)
                records.append({
                    "path": relative_path,
                    "label": class_index,
                    "class_name": class_name,
                    "width": int(width_height[0]),
                    "height": int(width_height[1]),
                    "mode": original_mode,
                    "pixel_sha256": content_hash,
                })
            except Exception as exc:
                corrupted.append(f"{relative_path}: {type(exc).__name__}: {exc}")

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    class_counts = {
        class_name: sum(1 for item in records if item["class_name"] == class_name)
        for class_name in CLASSES
    }
    info = {
        "dataset_folder_name": dataset_dir.name,
        "total_valid_images": len(records),
        "counts_by_class": class_counts,
        "dimensions_width_x_height": dict(size_counts),
        "original_pil_modes": dict(mode_counts),
        "expected_dimensions_width_x_height": list(EXPECTED_SIZE),
        "corrupted_files": corrupted,
        "exact_pixel_duplicates": duplicates,
        "imbalance_ratio_max_over_min": (
            round(max(class_counts.values()) / min(class_counts.values()), 4)
            if min(class_counts.values()) else None
        ),
    }
    return records, info


def make_stratified_splits(records: list[dict]) -> dict:
    paths = np.array([item["path"] for item in records])
    labels = np.array([item["label"] for item in records], dtype=np.int64)
    indices = np.arange(len(records))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED
    )

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    result = {
        "seed": SEED,
        "class_order": CLASSES,
        "proportions_requested": {"train": 0.70, "val": 0.15, "test": 0.15},
        "paths": {
            name: [str(paths[i]) for i in idx]
            for name, idx in split_indices.items()
        },
        "labels": {
            name: [int(labels[i]) for i in idx]
            for name, idx in split_indices.items()
        },
        "counts_by_split_and_class": {},
        "totals_by_split": {},
        "class_weights_from_training_only": {},
    }

    for split_name, idx in split_indices.items():
        split_counts = [int(np.sum(labels[idx] == class_index)) for class_index in range(len(CLASSES))]
        result["counts_by_split_and_class"][split_name] = dict(zip(CLASSES, split_counts))
        result["totals_by_split"][split_name] = int(len(idx))

    train_labels = labels[train_idx]
    train_counts = np.bincount(train_labels, minlength=len(CLASSES))
    if np.any(train_counts == 0):
        raise ValueError("A divisão deixou uma classe sem exemplos no treinamento.")
    total_train = int(len(train_labels))
    weights = total_train / (len(CLASSES) * train_counts)
    result["class_weights_from_training_only"] = {
        name: float(weights[index]) for index, name in enumerate(CLASSES)
    }

    # Verificação explícita: nenhuma imagem pode aparecer em mais de um subconjunto.
    sets = {name: set(values) for name, values in result["paths"].items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("Foi detectada sobreposição entre treino, validação e teste.")
    return result


def create_figures(dataset_dir: Path, records: list[dict], splits: dict) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    class_counts = splits["counts_by_split_and_class"]
    x = np.arange(len(CLASSES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, split_name, label in [(-width, "train", "Treino (70%)"), (0, "val", "Validação (15%)"), (width, "test", "Teste (15%)")]:
        values = [class_counts[split_name][name] for name in CLASSES]
        bars = ax.bar(x + offset, values, width, label=label)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xticks(x, LABELS_PT)
    ax.set_ylabel("Quantidade de imagens")
    ax.set_title("Distribuição por classe nos três subconjuntos")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_distribuicao.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 6, figsize=(13, 5))
    for class_index, class_name in enumerate(CLASSES):
        class_records = [item for item in records if item["class_name"] == class_name]
        for row in range(2):
            ax = axes[row, class_index]
            with Image.open(dataset_dir / class_records[row]["path"]) as image:
                ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{class_name}\n(n={len(class_records)})", fontsize=8)
    fig.suptitle("Duas imagens de exemplo por classe", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_exemplos.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    glass_records = [item for item in records if item["class_name"] == "glass"]
    sample_path = dataset_dir / glass_records[0]["path"]
    with Image.open(sample_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, (width_px, height_px) in zip(axes, [(128, 96), (256, 192), (512, 384)]):
            resized = original.resize((width_px, height_px), Image.Resampling.BILINEAR)
            ax.imshow(resized)
            ax.set_title(f"{width_px} × {height_px} px")
            ax.axis("off")
        fig.suptitle("A mesma imagem nas três resoluções")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_resolucoes.png", dpi=180)
        plt.close(fig)

    weights = splits["class_weights_from_training_only"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(LABELS_PT, [weights[c] for c in CLASSES])
    ax.bar_label(bars, fmt="%.3f", padding=2)
    ax.set_ylabel("Peso na função de perda")
    ax.set_title("Pesos inversamente proporcionais à frequência no treino")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_pesos_classe.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeção completa e divisão estratificada do TrashNet.")
    parser.add_argument("--no-download", action="store_true", help="Não baixar o ZIP automaticamente.")
    args = parser.parse_args()

    make_output_dirs()
    dataset_dir = find_dataset_dir(download_if_missing=not args.no_download)
    try:
        validate_reference_dataset_counts(dataset_dir)
    except Exception as exc:
        raise RuntimeError(f"Dataset inválido antes da análise: {exc}") from exc
    records, info = inspect_dataset(dataset_dir)

    if info["total_valid_images"] != 2527 or info["counts_by_class"] != EXPECTED_CLASS_COUNTS:
        save_json(info, METRICS_DIR / "inspecao_dataset.json")
        raise RuntimeError(
            "A inspeção não confirmou as 2.527 imagens e as contagens oficiais por classe. "
            "Consulte outputs/metricas/inspecao_dataset.json."
        )
    if info["corrupted_files"]:
        save_json(info, METRICS_DIR / "inspecao_dataset.json")
        raise RuntimeError(
            f"Foram encontradas {len(info['corrupted_files'])} imagens corrompidas. "
            "Consulte outputs/metricas/inspecao_dataset.json antes de continuar."
        )
    if info["exact_pixel_duplicates"]:
        save_json(info, METRICS_DIR / "inspecao_dataset.json")
        raise RuntimeError(
            f"Foram encontrados {len(info['exact_pixel_duplicates'])} grupos de imagens duplicadas. "
            "Revise os arquivos para evitar vazamento entre subconjuntos."
        )
    if not all((item["width"], item["height"]) == EXPECTED_SIZE for item in records):
        save_json(info, METRICS_DIR / "inspecao_dataset.json")
        raise RuntimeError(
            "Há imagens que não estão em 512 × 384 (largura × altura). "
            "Confira outputs/metricas/inspecao_dataset.json e confirme se foi usado o ZIP de referência."
        )

    splits = make_stratified_splits(records)
    info["split_counts"] = splits["counts_by_split_and_class"]
    info["split_totals"] = splits["totals_by_split"]
    info["dataset_exploration_summary"] = {
        "images_validated": len(records),
        "all_files_opened_successfully": True,
        "exact_duplicates_found": 0,
        "all_images_match_expected_size": all(
            (item["width"], item["height"]) == EXPECTED_SIZE for item in records
        ),
        "all_images_rgb_after_pipeline_conversion": True,
    }

    save_json(info, METRICS_DIR / "inspecao_dataset.json")
    save_json(splits, OUTPUT_DIR / "splits.json")
    create_figures(dataset_dir, records, splits)

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("\nDivisão criada:")
    print(json.dumps(splits["counts_by_split_and_class"], ensure_ascii=False, indent=2))
    print("\nArquivos de saída: outputs/splits.json, outputs/metricas/inspecao_dataset.json e outputs/figuras/.")


if __name__ == "__main__":
    main()
