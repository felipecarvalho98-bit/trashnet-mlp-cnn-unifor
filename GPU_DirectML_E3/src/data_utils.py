from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .config import CLASSES, DATASET_ZIP, EXTRACT_DIR, OUTPUT_DIR, SEED, SPLITS_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_COUNTS = {
    "glass": 501,
    "paper": 594,
    "cardboard": 403,
    "plastic": 480,
    "metal": 409,
    "trash": 137,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Caminho inseguro encontrado no ZIP: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Arquivo fora da pasta de destino: {member.filename}")
        archive.extractall(destination)


def locate_dataset() -> Path:
    """Extrai o ZIP que acompanha o pacote e localiza as seis pastas de classe."""
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {DATASET_ZIP}. Recoloque o ZIP original dentro da pasta data."
        )

    for candidate in [EXTRACT_DIR, *EXTRACT_DIR.glob("**/*")]:
        if candidate.is_dir() and all((candidate / name).is_dir() for name in CLASSES):
            return candidate

    safe_extract(DATASET_ZIP, EXTRACT_DIR)
    for candidate in [EXTRACT_DIR, *EXTRACT_DIR.glob("**/*")]:
        if candidate.is_dir() and all((candidate / name).is_dir() for name in CLASSES):
            return candidate
    raise FileNotFoundError("Não encontrei as seis pastas de classe dentro do ZIP do TrashNet.")


def scan_and_split(dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    """Lê as imagens, exclui cópias binárias exatas e cria a divisão 70/15/15."""
    records: list[dict] = []
    invalid: list[dict] = []
    duplicates: list[dict] = []
    hash_seen: dict[str, dict] = {}

    # A ordem de classes é intencional: em conflitos de duplicatas, a cópia de glass é mantida.
    for class_index, class_name in enumerate(CLASSES):
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Pasta ausente: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    width, height = image.size
                    if width < 2 or height < 2:
                        raise ValueError("Dimensões inválidas")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid.append({"path": path.relative_to(dataset_root).as_posix(), "reason": str(exc)})
                continue

            file_hash = sha256_file(path)
            relative = path.relative_to(dataset_root).as_posix()
            row = {
                "relative_path": relative,
                "class_name": class_name,
                "class_index": class_index,
                "width": width,
                "height": height,
                "sha256": file_hash,
            }
            if file_hash in hash_seen:
                first = hash_seen[file_hash]
                duplicates.append({
                    "duplicate_path": relative,
                    "kept_path": first["relative_path"],
                    "duplicate_class": class_name,
                    "kept_class": first["class_name"],
                    "same_class": class_name == first["class_name"],
                    "sha256": file_hash,
                })
                continue
            hash_seen[file_hash] = row
            records.append(row)

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Nenhuma imagem válida foi encontrada.")

    counts = frame.groupby("class_name").size().reindex(CLASSES, fill_value=0).to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            "A contagem do dataset não corresponde à base já usada no projeto. "
            f"Esperado: {EXPECTED_COUNTS}; encontrado: {counts}. "
            f"Duplicatas detectadas: {len(duplicates)}."
        )

    # Primeiro reservamos 15% para teste. Depois retiramos 15% do total para validação
    # entre os 85% restantes, usando exatamente a semente 42.
    train_val, test = train_test_split(
        frame,
        test_size=0.15,
        random_state=SEED,
        stratify=frame["class_index"],
    )
    validation_relative = 0.15 / (0.70 + 0.15)
    train, validation = train_test_split(
        train_val,
        test_size=validation_relative,
        random_state=SEED,
        stratify=train_val["class_index"],
    )

    pieces = []
    for split_name, part in (("train", train), ("validation", validation), ("test", test)):
        copy = part.copy()
        copy["split"] = split_name
        pieces.append(copy)
    splits = pd.concat(pieces, ignore_index=True)
    splits = splits.sort_values(["split", "class_index", "relative_path"]).reset_index(drop=True)

    # Garante que a mesma imagem não foi colocada em partições diferentes.
    if splits.groupby("sha256")["split"].nunique().max() > 1:
        raise RuntimeError("Foi encontrada duplicata cruzando as partições.")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_csv(SPLITS_DIR / "splits.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "dataset_root": str(dataset_root),
        "dataset_zip": DATASET_ZIP.name,
        "dataset_zip_sha256": sha256_file(DATASET_ZIP),
        "valid_images_after_exact_deduplication": int(len(frame)),
        "class_order": list(CLASSES),
        "counts_by_class": counts,
        "removed_exact_duplicates": duplicates,
        "invalid_images": invalid,
        "seed": SEED,
        "split_sizes": splits.groupby("split").size().to_dict(),
        "split_class_counts": {
            split_name: splits[splits.split == split_name].groupby("class_name").size().reindex(CLASSES, fill_value=0).to_dict()
            for split_name in ("train", "validation", "test")
        },
        "preprocessing": "RGB; redimensionamento para 128x96; pixels convertidos para float32 em [0,1]",
        "augmentation": "Não utilizada",
    }
    (SPLITS_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return splits, metadata


def load_or_prepare() -> tuple[pd.DataFrame, Path, dict]:
    dataset_root = locate_dataset()
    frame, metadata = scan_and_split(dataset_root)
    return frame, dataset_root, metadata
