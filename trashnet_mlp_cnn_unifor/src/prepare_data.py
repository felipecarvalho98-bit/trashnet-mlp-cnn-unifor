"""Preparação e análise inicial dos dados."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

from .config import CLASS_NAMES, OUTPUT_DIR, SEED, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION
from .data_utils import locate_dataset, scan_dataset, create_splits


def _save_initial_figures(df: pd.DataFrame, splits: pd.DataFrame, out_dir: Path) -> None:
    "Gera os gráficos que ajudam a conhecer os dados antes do treinamento: distribuição por partição, exemplos de cada classe, comparação de resoluções e pesos calculados para as classes."
    out_dir.mkdir(parents=True, exist_ok=True)
    # Figura 1: quantidade por classe em cada partição.
    pivot = splits.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(CLASS_NAMES)
    ax = pivot[["train", "validation", "test"]].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Distribuição de imagens por classe e partição")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Quantidade de imagens")
    ax.legend(title="Partição")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_distribuicao_particoes.png", dpi=160)
    plt.close()

    # Figura 2: ao menos duas imagens por classe.
    fig, axes = plt.subplots(len(CLASS_NAMES), 2, figsize=(7, 15))
    for row, class_name in enumerate(CLASS_NAMES):
        examples = df[df.class_name == class_name].head(2)
        for col in range(2):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(examples):
                record = examples.iloc[col]
                with Image.open(record.path) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{class_name} | {record.width}×{record.height}")
    fig.suptitle("Exemplos do conjunto TrashNet (duas imagens por classe)", y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_exemplos_por_classe.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Figura 3: a MESMA imagem redimensionada em três resoluções.
    sample = df.iloc[0]
    with Image.open(sample.path) as im:
        rgb = im.convert("RGB")
        sizes = [(128, 96), (256, 192), (512, 384)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, size in zip(axes, sizes):
            ax.imshow(rgb.resize(size, Image.Resampling.LANCZOS))
            ax.set_title(f"{size[0]} × {size[1]}")
            ax.axis("off")
        fig.suptitle(f"Mesma imagem em três resoluções — classe {sample.class_name}")
        plt.tight_layout()
        plt.savefig(out_dir / "fig3_comparacao_resolucoes.png", dpi=160)
        plt.close()

    # Figura 4: pesos que seriam usados nos experimentos com pesos de classe.
    train_counts = splits[splits.split == "train"].groupby("class_index").size().reindex(range(len(CLASS_NAMES)), fill_value=0)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=splits.loc[splits.split == "train", "class_index"].to_numpy())
    weight_df = pd.DataFrame({"class_name": CLASS_NAMES, "train_count": train_counts.to_numpy(), "class_weight": weights})
    weight_df.to_csv(out_dir / "pesos_de_classe.csv", index=False)
    ax = weight_df.plot(x="class_name", y="class_weight", kind="bar", legend=False, figsize=(8, 4))
    ax.set_title("Pesos de classe calculados somente com o treino")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Peso")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_pesos_de_classe.png", dpi=160)
    plt.close()


def main() -> None:
    "Organiza a preparação inicial: localiza e verifica as imagens, remove duplicatas exatas, cria as partições, salva os metadados e gera os gráficos."
    dataset_root = locate_dataset()
    print(f"Dataset localizado em: {dataset_root}")
    df, duplicates, scan_info = scan_dataset(dataset_root)
    splits = create_splits(df)
    out_dir = OUTPUT_DIR / "preparacao"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out_dir / "splits.csv", index=False)
    duplicates.to_csv(out_dir / "duplicatas_exatas.csv", index=False)
    pd.DataFrame(scan_info["invalid_images"]).to_csv(out_dir / "imagens_invalidas.csv", index=False)

    counts = splits.groupby(["split", "class_name"]).size().unstack(fill_value=0).reindex(columns=CLASS_NAMES, fill_value=0)
    counts.to_csv(out_dir / "contagem_por_particao.csv")
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "total_images_after_exact_deduplication": int(len(df)),
        "class_count": int(df.class_name.nunique()),
        "class_names": list(CLASS_NAMES),
        "counts_by_class": {k: int(v) for k, v in df.class_name.value_counts().reindex(CLASS_NAMES, fill_value=0).items()},
        "split_counts": {k: int((splits.split == k).sum()) for k in ["train", "validation", "test"]},
        "split_percentages": {k: float((splits.split == k).mean() * 100) for k in ["train", "validation", "test"]},
        "seed": SEED,
        "requested_split": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION, "test": TEST_FRACTION},
        "deduplication": {
            "exact_duplicates_removed": int(scan_info["duplicate_count"]),
            "duplicate_label_conflicts": int(scan_info["duplicate_label_conflicts"]),
            "invalid_images": int(scan_info["invalid_count"]),
        },
        "note": "Duplicatas binárias exatas são identificadas por SHA-256; cópias repetidas são removidas antes da divisão. Duplicatas visualmente semelhantes não são detectadas automaticamente.",
    }
    (out_dir / "dataset_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_initial_figures(df, splits, out_dir)

    print("\nPreparação concluída.")
    print(f"Total de imagens válidas após deduplicação: {len(df)}")
    print("\nContagem por partição:")
    print(counts)
    print(f"\nArquivos salvos em: {out_dir}")
    if scan_info["duplicate_label_conflicts"]:
        print(f"ATENÇÃO: {scan_info['duplicate_label_conflicts']} duplicatas exatas estavam em pastas de classes diferentes. Confira duplicatas_exatas.csv.")


if __name__ == "__main__":
    main()
