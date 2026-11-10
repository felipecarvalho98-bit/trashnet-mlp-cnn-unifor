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


def _class_split_quotas(labels: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Calcula cotas estratificadas por classe, preservando os totais 70/15/15."""
    total = int(len(labels))
    test_pool = int(np.ceil(0.30 * total))
    train_total = total - test_pool
    val_total = test_pool - int(np.ceil(0.50 * test_pool))
    test_total = int(np.ceil(0.50 * test_pool))
    split_totals = [train_total, val_total, test_total]
    proportions = np.array([0.70, 0.15, 0.15], dtype=float)

    class_counts = np.bincount(labels, minlength=len(CLASSES)).astype(int)
    exact = class_counts[:, None] * proportions[None, :]
    quotas = np.floor(exact).astype(int)
    row_remaining = class_counts - quotas.sum(axis=1)
    col_remaining = np.array(split_totals, dtype=int) - quotas.sum(axis=0)
    fractions = exact - quotas
    used_cells: set[tuple[int, int]] = set()

    # Cada classe recebe apenas os arredondamentos que faltam; cada total de
    # subconjunto também é fechado exatamente. Empates são resolvidos pela ordem fixa.
    while int(row_remaining.sum()) > 0:
        candidates = [
            (float(fractions[class_index, split_index]), -class_index, -split_index,
             class_index, split_index)
            for class_index in range(len(CLASSES))
            if row_remaining[class_index] > 0
            for split_index in range(3)
            if col_remaining[split_index] > 0
            and (class_index, split_index) not in used_cells
        ]
        if not candidates:
            raise RuntimeError("Não foi possível fechar as cotas estratificadas 70/15/15.")
        _, _, _, class_index, split_index = max(candidates)
        quotas[class_index, split_index] += 1
        row_remaining[class_index] -= 1
        col_remaining[split_index] -= 1
        used_cells.add((class_index, split_index))

    if np.any(col_remaining != 0):
        raise RuntimeError(f"Cotas totais inconsistentes: {col_remaining.tolist()}")
    totals = {"train": train_total, "val": val_total, "test": test_total}
    return quotas, totals


def make_stratified_splits(records: list[dict]) -> dict:
    """Divide dados estratificados e mantém hashes de pixels duplicados no mesmo split."""
    if not records:
        raise ValueError("Nenhum registro foi fornecido para a divisão.")

    paths = np.array([item["path"] for item in records])
    labels = np.array([item["label"] for item in records], dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= len(CLASSES)):
        raise ValueError("Há rótulos fora do intervalo das seis classes.")

    quotas, expected_totals = _class_split_quotas(labels)
    # quotas tem formato classe x subconjunto.
    proportions = np.array([0.70, 0.15, 0.15], dtype=float)

    hash_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        pixel_hash = item.get("pixel_sha256")
        # Em testes/unitários sem hash, cada caminho é tratado como imagem única.
        group_key = f"hash:{pixel_hash}" if pixel_hash else f"path:{item['path']}"
        hash_to_indices[group_key].append(index)

    duplicate_groups = [
        sorted(indices, key=lambda idx: str(paths[idx]))
        for _, indices in sorted(hash_to_indices.items())
        if len(indices) > 1
    ]
    singleton_indices = [
        indices[0] for _, indices in sorted(hash_to_indices.items())
        if len(indices) == 1
    ]

    group_class_counts = [
        np.bincount(labels[group], minlength=len(CLASSES)).astype(int)
        for group in duplicate_groups
    ]
    duplicate_totals_by_class = (
        np.sum(group_class_counts, axis=0)
        if group_class_counts else np.zeros(len(CLASSES), dtype=int)
    )
    target_duplicate_counts = duplicate_totals_by_class[:, None] * proportions[None, :]

    best_assignment = None
    best_cost = float("inf")

    if len(duplicate_groups) <= 12:
        # São poucos grupos no dataset de referência; a busca exaustiva torna a
        # escolha determinística e respeita os limites de cada classe/subconjunto.
        from itertools import product
        for assignment in product(range(3), repeat=len(duplicate_groups)):
            allocated = np.zeros((len(CLASSES), 3), dtype=int)
            for group_index, split_index in enumerate(assignment):
                allocated[:, split_index] += group_class_counts[group_index]
            if np.any(allocated > quotas):
                continue
            cost = float(np.sum(
                ((allocated - target_duplicate_counts) ** 2)
                / (target_duplicate_counts + 1.0)
            ))
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment
    else:
        # Fallback escalável para datasets com muitos grupos duplicados.
        allocated = np.zeros((len(CLASSES), 3), dtype=int)
        assignment = []
        order = sorted(
            range(len(duplicate_groups)),
            key=lambda idx: (-len(duplicate_groups[idx]),
                             tuple(str(paths[i]) for i in duplicate_groups[idx]))
        )
        local_assignment = {}
        for group_index in order:
            group_counts = group_class_counts[group_index]
            options = []
            for split_index in range(3):
                candidate = allocated.copy()
                candidate[:, split_index] += group_counts
                if np.any(candidate > quotas):
                    continue
                cost = float(np.sum(
                    ((candidate - target_duplicate_counts) ** 2)
                    / (target_duplicate_counts + 1.0)
                ))
                options.append((cost, split_index, candidate))
            if not options:
                raise RuntimeError(
                    "Um grupo de imagens duplicadas é maior que as cotas disponíveis. "
                    "Revise os grupos no relatório de inspeção."
                )
            cost, split_index, allocated = min(options, key=lambda item: (item[0], item[1]))
            local_assignment[group_index] = split_index
        best_assignment = tuple(local_assignment[i] for i in range(len(duplicate_groups)))

    if best_assignment is None:
        raise RuntimeError(
            "Não foi possível alocar grupos duplicados sem ultrapassar as cotas por classe. "
            "Consulte outputs/metricas/inspecao_dataset.json."
        )

    split_names = ["train", "val", "test"]
    split_indices: dict[str, list[int]] = {name: [] for name in split_names}
    duplicate_group_paths = []
    duplicate_group_splits = []
    allocated = np.zeros((len(CLASSES), 3), dtype=int)

    for group_index, group in enumerate(duplicate_groups):
        split_index = int(best_assignment[group_index])
        split_name = split_names[split_index]
        split_indices[split_name].extend(group)
        allocated[:, split_index] += group_class_counts[group_index]
        duplicate_group_paths.append([str(paths[idx]) for idx in group])
        duplicate_group_splits.append(split_name)

    rng = np.random.default_rng(SEED)
    for class_index, class_name in enumerate(CLASSES):
        available = np.array(
            [idx for idx in singleton_indices if labels[idx] == class_index],
            dtype=int,
        )
        rng.shuffle(available)
        cursor = 0
        for split_index, split_name in enumerate(split_names):
            needed = int(quotas[class_index, split_index] - allocated[class_index, split_index])
            if needed < 0:
                raise RuntimeError(
                    f"Os grupos duplicados excedem a cota da classe {class_name} em {split_name}."
                )
            chosen = available[cursor:cursor + needed].tolist()
            split_indices[split_name].extend(chosen)
            cursor += needed
        if cursor != len(available):
            raise RuntimeError(
                f"Não foi possível distribuir todas as imagens únicas da classe {class_name}: "
                f"{cursor} de {len(available)}."
            )

    result = {
        "seed": SEED,
        "class_order": CLASSES,
        "proportions_requested": {"train": 0.70, "val": 0.15, "test": 0.15},
        "paths": {
            name: [str(paths[i]) for i in split_indices[name]]
            for name in split_names
        },
        "labels": {
            name: [int(labels[i]) for i in split_indices[name]]
            for name in split_names
        },
        "counts_by_split_and_class": {},
        "totals_by_split": {},
        "class_weights_from_training_only": {},
        "duplicate_groups": duplicate_group_paths,
        "duplicate_group_splits": duplicate_group_splits,
        "duplicate_group_policy": (
            "Todas as imagens com hash de pixels idêntico são mantidas no dataset "
            "e atribuídas integralmente ao mesmo subconjunto."
        ),
    }

    for split_index, split_name in enumerate(split_names):
        split_counts = [
            int(np.sum(labels[split_indices[split_name]] == class_index))
            for class_index in range(len(CLASSES))
        ]
        result["counts_by_split_and_class"][split_name] = dict(zip(CLASSES, split_counts))
        result["totals_by_split"][split_name] = int(len(split_indices[split_name]))

    train_labels = labels[split_indices["train"]]
    train_counts = np.bincount(train_labels, minlength=len(CLASSES))
    if np.any(train_counts == 0):
        raise ValueError("A divisão deixou uma classe sem exemplos no treinamento.")
    total_train = int(len(train_labels))
    weights = total_train / (len(CLASSES) * train_counts)
    result["class_weights_from_training_only"] = {
        name: float(weights[index]) for index, name in enumerate(CLASSES)
    }

    # Valida que nenhum caminho nem grupo de pixels aparece em splits distintos.
    path_to_split = {
        path: split_name
        for split_name in split_names
        for path in result["paths"][split_name]
    }
    if len(path_to_split) != len(records):
        raise RuntimeError("Há caminhos repetidos ou ausentes na divisão.")
    for group in duplicate_group_paths:
        assigned_splits = {path_to_split[path] for path in group}
        if len(assigned_splits) != 1:
            raise RuntimeError(f"Um grupo duplicado foi dividido entre subconjuntos: {group}")

    if result["totals_by_split"] != expected_totals:
        raise RuntimeError(
            f"Totais da divisão incorretos: {result['totals_by_split']} != {expected_totals}"
        )
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
    record_by_path = {item["path"]: item for item in records}
    info["exact_pixel_duplicate_details"] = [
        {
            "paths": group,
            "classes": [record_by_path[path]["class_name"] for path in group],
            "same_class": len({record_by_path[path]["class_name"] for path in group}) == 1,
        }
        for group in info["exact_pixel_duplicates"]
    ]

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
        print(
            f"ATENÇÃO: encontrados {len(info['exact_pixel_duplicates'])} grupos de pixels idênticos. "
            "Nenhuma imagem será removida; cada grupo ficará integralmente em um único subconjunto."
        )
        for detail in info["exact_pixel_duplicate_details"]:
            print("Grupo duplicado:", detail)
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
        "exact_duplicate_groups_found": len(info["exact_pixel_duplicates"]),
        "duplicate_groups_kept_within_one_split": True,
        "all_images_match_expected_size": all(
            (item["width"], item["height"]) == EXPECTED_SIZE for item in records
        ),
        "all_images_rgb_after_pipeline_conversion": True,
    }

    save_json(info, METRICS_DIR / "inspecao_dataset.json")
    save_json(splits, OUTPUT_DIR / "splits.json")
    create_figures(dataset_dir, records, splits)

    print(json.dumps(info, ensure_ascii=False, indent=2))
    if splits["duplicate_groups"]:
        print("\nGrupos duplicados mantidos no mesmo subconjunto:")
        for group, split_name in zip(splits["duplicate_groups"], splits["duplicate_group_splits"]):
            print(f" - {split_name}: {group}")
    else:
        print("\nNenhum grupo de pixels idênticos foi encontrado.")

    print("\nDivisão criada:")
    print(json.dumps(splits["counts_by_split_and_class"], ensure_ascii=False, indent=2))
    print("\nArquivos de saída: outputs/splits.json, outputs/metricas/inspecao_dataset.json e outputs/figuras/.")


if __name__ == "__main__":
    main()
