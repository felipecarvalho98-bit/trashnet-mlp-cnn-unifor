"""Funções de leitura e preparação do TrashNet."""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import (
    CLASS_NAMES, CLASS_TO_INDEX, DATA_DIR, IMAGE_EXTENSIONS,
    ROOT_DIR, SEED, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION,
    OUTPUT_DIR,
)


def _has_class_folders(path: Path) -> bool:
    "Confere se uma pasta contém as seis categorias do TrashNet. Isso evita começar a preparação apontando para uma pasta errada."
    if not path.is_dir():
        return False
    children = {p.name.lower(): p for p in path.iterdir() if p.is_dir()}
    return all(name in children for name in CLASS_NAMES)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extrai ZIP sem permitir escrita fora da pasta de destino."""
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Caminho inseguro encontrado no ZIP: {member.filename}")
        zf.extractall(destination)


def _walk_for_dataset(search_roots: Iterable[Path], max_depth: int = 5) -> Path | None:
    "Procura as pastas das classes em alguns níveis abaixo do caminho informado. Isso ajuda quando o dataset vem dentro de uma pasta extra, como acontece em alguns ZIPs."
    seen = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if _has_class_folders(root):
            return root
        # Procura diretórios até alguns níveis abaixo, evitando varrer arquivos de imagem.
        frontier = [(root, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            try:
                subdirs = [p for p in current.iterdir() if p.is_dir()]
            except OSError:
                continue
            for child in subdirs:
                if _has_class_folders(child):
                    return child.resolve()
                # Não desce em pastas típicas de ambiente/saída.
                if child.name.lower() not in {".venv", "venv", "outputs", "__pycache__", ".git"}:
                    frontier.append((child, depth + 1))
    return None


def locate_dataset() -> Path:
    """Localiza dataset extraído ou extrai um ZIP colocado dentro de data/."""
    override = os.environ.get("TRASHNET_DATA_DIR")
    candidates = []
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file() and override_path.suffix.lower() == ".zip":
            extracted = DATA_DIR / "raw" / "dataset_extraido"
            _safe_extract(override_path, extracted)
            found = _walk_for_dataset([extracted])
            if found:
                return found
        candidates.append(override_path)
    candidates.extend([
        DATA_DIR / "dataset-resized",
        DATA_DIR / "raw" / "dataset-resized",
        DATA_DIR / "raw",
        DATA_DIR,
        ROOT_DIR / "data",
    ])
    found = _walk_for_dataset(candidates)
    if found:
        return found

    # Procura ZIPs no diretório data e extrai o primeiro que contenha as seis classes.
    zip_candidates = sorted(DATA_DIR.glob("*.zip")) + sorted((DATA_DIR / "raw").glob("*.zip")) if (DATA_DIR / "raw").exists() else sorted(DATA_DIR.glob("*.zip"))
    for zip_path in zip_candidates:
        destination = DATA_DIR / "raw" / f"extraido_{zip_path.stem}"
        if not destination.exists() or not any(destination.iterdir()):
            _safe_extract(zip_path, destination)
        found = _walk_for_dataset([destination])
        if found:
            return found

    raise FileNotFoundError(
        "Não encontrei as seis pastas de classes do TrashNet. Coloque a pasta extraída em "
        f"'{DATA_DIR / 'dataset-resized'}' ou o ZIP dentro de '{DATA_DIR}'. "
        "As subpastas esperadas são: " + ", ".join(CLASS_NAMES) + "."
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    "Gera uma assinatura do conteúdo do arquivo. Usamos essa assinatura para reconhecer cópias exatamente iguais, mesmo que tenham nomes diferentes."
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lista imagens válidas e remove cópias exatas mantendo a primeira ocorrência."""
    records = []
    invalid = []
    duplicates = []
    hash_seen: dict[str, dict] = {}
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            # tolera diferença apenas de maiúsculas/minúsculas
            match = next((p for p in dataset_root.iterdir() if p.is_dir() and p.name.lower() == class_name), None)
            if match is None:
                raise FileNotFoundError(f"Pasta da classe ausente: {class_name} em {dataset_root}")
            class_dir = match
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    width, height = im.size
                    if width < 2 or height < 2:
                        raise ValueError("Imagem com dimensões inválidas")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid.append({"path": str(path), "reason": str(exc)})
                continue
            file_hash = sha256_file(path)
            record = {
                "path": str(path.resolve()),
                "relative_path": path.resolve().relative_to(dataset_root.resolve()).as_posix(),
                "class_name": class_name,
                "class_index": CLASS_TO_INDEX[class_name],
                "width": width,
                "height": height,
                "sha256": file_hash,
            }
            if file_hash in hash_seen:
                first = hash_seen[file_hash]
                duplicates.append({
                    "duplicate_path": record["relative_path"],
                    "kept_path": first["relative_path"],
                    "duplicate_class": class_name,
                    "kept_class": first["class_name"],
                    "same_class": class_name == first["class_name"],
                    "sha256": file_hash,
                })
                continue
            hash_seen[file_hash] = record
            records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"Nenhuma imagem válida encontrada em {dataset_root}.")
    found_classes = set(df["class_name"])
    missing_classes = set(CLASS_NAMES) - found_classes
    if missing_classes:
        raise ValueError(f"Classes sem imagens válidas: {sorted(missing_classes)}")
    return df, pd.DataFrame(duplicates), {
        "invalid_images": invalid,
        "duplicate_count": len(duplicates),
        "duplicate_label_conflicts": sum(not d["same_class"] for d in duplicates),
        "invalid_count": len(invalid),
    }


def create_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Cria split estratificado 70/15/15 com seed fixa e sem duplicatas exatas."""
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df, test_size=TEST_FRACTION, random_state=SEED,
        stratify=df["class_index"],
    )
    # 15% do total representa 15/85 dos dados restantes.
    validation_relative = VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)
    train, validation = train_test_split(
        train_val, test_size=validation_relative, random_state=SEED,
        stratify=train_val["class_index"],
    )
    parts = []
    for split_name, part in (("train", train), ("validation", validation), ("test", test)):
        piece = part.copy()
        piece["split"] = split_name
        parts.append(piece)
    result = pd.concat(parts, ignore_index=True)
    # Verificação explícita: nenhum hash pode aparecer em mais de uma partição.
    overlap = result.groupby("sha256")["split"].nunique().max()
    if overlap > 1:
        raise RuntimeError("Foi detectada duplicata entre partições.")
    return result.sort_values(["split", "class_index", "relative_path"]).reset_index(drop=True)


def load_splits() -> tuple[pd.DataFrame, Path]:
    "Lê a divisão já criada e reconstrói os caminhos das imagens no computador atual. Dessa forma, o treinamento usa a mesma lista de imagens sem precisar criar outra divisão."
    split_path = OUTPUT_DIR / "preparacao" / "splits.csv"
    metadata_path = OUTPUT_DIR / "preparacao" / "dataset_metadata.json"
    if not split_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Partições não encontradas. Execute primeiro: python -m src.prepare_data")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    override = os.environ.get("TRASHNET_DATA_DIR")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            candidate = locate_dataset()
        dataset_root = _walk_for_dataset([candidate]) or candidate
    else:
        previous = Path(metadata.get("dataset_root", ""))
        dataset_root = previous if _has_class_folders(previous) else locate_dataset()
    df = pd.read_csv(split_path)
    df["path"] = df["relative_path"].apply(lambda p: str((dataset_root / p).resolve()))
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imagens referenciadas no split não foram encontradas. Exemplo: {missing[0]}")
    return df, dataset_root
