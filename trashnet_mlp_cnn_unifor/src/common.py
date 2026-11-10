from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Os diretórios podem ser redirecionados por variáveis de ambiente no Colab.
# Assim, os dados ficam na VM local (leitura rápida) e os resultados persistem no Drive.
DATA_DIR = Path(os.environ.get("TRASHNET_DATA_DIR", ROOT / "data")).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("TRASHNET_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
FIG_DIR = OUTPUT_DIR / "figuras"
METRICS_DIR = OUTPUT_DIR / "metricas"
HISTORY_DIR = OUTPUT_DIR / "historicos"
MODELS_DIR = OUTPUT_DIR / "modelos"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

# A ordem é fixa e segue exatamente a proposta do professor.
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
LABELS_PT = ["Vidro", "Papel", "Papelão", "Plástico", "Metal", "Outros"]
SEED = 42

# Quantidades indicadas na proposta do professor para dataset-resized.zip.
EXPECTED_CLASS_COUNTS = {
    "glass": 501,
    "paper": 594,
    "cardboard": 403,
    "plastic": 482,
    "metal": 410,
    "trash": 137,
}
EXPECTED_TOTAL_IMAGES = sum(EXPECTED_CLASS_COUNTS.values())

# Cotas inteiras para a divisão 70/15/15 da semente 42 e 2.527 imagens.
# Os totais são 1.768 treino, 379 validação e 380 teste.
EXPECTED_SPLIT_CLASS_COUNTS = {
    "train": {"glass": 350, "paper": 416, "cardboard": 282, "plastic": 337, "metal": 287, "trash": 96},
    "val":   {"glass": 75, "paper": 89, "cardboard": 60, "plastic": 72, "metal": 62, "trash": 21},
    "test":  {"glass": 76, "paper": 89, "cardboard": 61, "plastic": 73, "metal": 61, "trash": 20},
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_IMAGE_SIZE = (512, 384)  # largura x altura

DATASET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/"
    "dataset-resized.zip"
)
DATASET_SHA256 = "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c"
DATASET_ZIP = DATA_DIR / "dataset-resized.zip"

EXPERIMENTS = {
    "E1": {"model": "mlp", "width": 128, "height": 96, "class_weights": False},
    "E2": {"model": "mlp", "width": 128, "height": 96, "class_weights": True},
    "E3": {"model": "cnn", "width": 128, "height": 96, "class_weights": False},
    "E4": {"model": "cnn", "width": 128, "height": 96, "class_weights": True},
    "E5": {"model": "cnn", "width": 256, "height": 192, "class_weights": False},
    "E6": {"model": "cnn", "width": 512, "height": 384, "class_weights": False},
}


def make_output_dirs() -> None:
    for path in (DATA_DIR, OUTPUT_DIR, FIG_DIR, METRICS_DIR, HISTORY_DIR, MODELS_DIR, CHECKPOINTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(force: bool = False) -> Path:
    """Baixa o ZIP oficial e valida o SHA-256 publicado pelo repositório do dataset."""
    make_output_dirs()
    if DATASET_ZIP.exists() and not force:
        actual = sha256_file(DATASET_ZIP)
        if actual == DATASET_SHA256:
            return DATASET_ZIP
        raise ValueError(
            f"O arquivo {DATASET_ZIP} já existe, mas o SHA-256 não confere. "
            "Remova-o ou execute com --force-download."
        )

    temporary = DATASET_ZIP.with_suffix(".zip.part")
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "TrashNet-MLP-CNN-course-project/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            total = int(response.headers.get("Content-Length", "0") or 0)
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                received += len(chunk)
                if total:
                    print(f"Download: {received / total:.0%}", end="\r")
        print()
        actual = sha256_file(temporary)
        if actual != DATASET_SHA256:
            raise ValueError(
                "O SHA-256 do arquivo baixado não corresponde ao arquivo oficial. "
                f"Esperado: {DATASET_SHA256}; recebido: {actual}."
            )
        temporary.replace(DATASET_ZIP)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return DATASET_ZIP


def _safe_extract(zip_path: Path, destination: Path) -> None:
    """Extrai o ZIP sem permitir que caminhos internos escapem da pasta de dados."""
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Caminho inseguro dentro do ZIP: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Arquivo fora da pasta de destino: {member.filename}")
        archive.extractall(destination)


def find_dataset_dir(download_if_missing: bool = True) -> Path:
    """Localiza a pasta com as seis classes, extraindo o ZIP se necessário."""
    make_output_dirs()

    candidates = [DATA_DIR / "dataset-resized"]
    candidates.extend(p for p in DATA_DIR.iterdir() if p.is_dir() and p not in candidates)
    for candidate in candidates:
        if all((candidate / class_name).is_dir() for class_name in CLASSES):
            return candidate

    zip_files = [DATASET_ZIP] + sorted(DATA_DIR.glob("*.zip"))
    zip_path = next((p for p in zip_files if p.is_file()), None)
    if zip_path is None and download_if_missing:
        zip_path = download_dataset()
    if zip_path is None:
        raise FileNotFoundError(
            "Dataset não encontrado. Coloque data/dataset-resized.zip ou permita o download automático."
        )

    _safe_extract(zip_path, DATA_DIR)
    candidates = [DATA_DIR / "dataset-resized"]
    candidates.extend(p for p in DATA_DIR.iterdir() if p.is_dir() and p not in candidates)
    for candidate in candidates:
        if all((candidate / class_name).is_dir() for class_name in CLASSES):
            return candidate
    raise FileNotFoundError(
        "O ZIP foi extraído, mas não localizei uma pasta contendo as seis classes: "
        + ", ".join(CLASSES)
    )



def validate_reference_dataset_counts(dataset_dir: Path) -> dict[str, int]:
    """Impede que uma pasta de amostras/testes seja usada como TrashNet oficial."""
    actual = {}
    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Pasta da classe ausente: {class_dir}")
        actual[class_name] = sum(
            1 for file_path in class_dir.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if actual != EXPECTED_CLASS_COUNTS:
        differences = {
            name: {"esperado": EXPECTED_CLASS_COUNTS[name], "encontrado": actual[name]}
            for name in CLASSES if actual[name] != EXPECTED_CLASS_COUNTS[name]
        }
        raise ValueError(
            "A pasta encontrada não corresponde ao dataset-resized.zip de referência "
            f"(2.527 imagens). Diferenças por classe: {json.dumps(differences, ensure_ascii=False)}. "
            "Remova a pasta de teste e execute novamente para baixar/usar o ZIP oficial."
        )
    return actual


def validate_splits(splits: dict[str, Any]) -> None:
    """Confere totais, estratificação, integridade e grupos duplicados sem vazamento."""
    if splits.get("class_order") != CLASSES:
        raise ValueError("A ordem de classes em outputs/splits.json não corresponde à proposta.")

    paths = splits.get("paths", {})
    labels = splits.get("labels", {})
    expected_sizes = {"train": 1768, "val": 379, "test": 380}
    all_paths = []
    total_by_class = {name: 0 for name in CLASSES}
    path_to_split = {}

    for split_name, expected_size in expected_sizes.items():
        split_paths = paths.get(split_name, [])
        split_labels = labels.get(split_name, [])
        if len(split_paths) != expected_size or len(split_labels) != expected_size:
            raise ValueError(
                f"Divisão {split_name} inválida: esperados {expected_size} exemplos; "
                f"encontrados {len(split_paths)} caminhos e {len(split_labels)} rótulos. "
                "Execute primeiro python -m src.prepare_data."
            )
        if len(split_paths) != len(set(split_paths)):
            raise ValueError(f"Há caminhos repetidos dentro do conjunto {split_name}.")

        split_counts = {name: 0 for name in CLASSES}
        for relative_path, label in zip(split_paths, split_labels):
            if not isinstance(label, int) or label < 0 or label >= len(CLASSES):
                raise ValueError(f"Rótulo inválido encontrado no conjunto {split_name}: {label}")
            class_from_path = PurePosixPath(str(relative_path)).parts[0]
            if class_from_path != CLASSES[label]:
                raise ValueError(
                    f"Caminho e rótulo incompatíveis em {split_name}: {relative_path} -> {CLASSES[label]}"
                )
            total_by_class[CLASSES[label]] += 1
            split_counts[CLASSES[label]] += 1
            path_to_split[str(relative_path)] = split_name

        expected_counts = EXPECTED_SPLIT_CLASS_COUNTS[split_name]
        if split_counts != expected_counts:
            raise ValueError(
                f"A divisão não corresponde às cotas estratificadas esperadas em {split_name}: "
                f"encontrado {split_counts}; esperado {expected_counts}. "
                "Execute novamente python -m src.prepare_data."
            )
        all_paths.extend(split_paths)

    if len(all_paths) != EXPECTED_TOTAL_IMAGES or len(set(all_paths)) != EXPECTED_TOTAL_IMAGES:
        raise ValueError("A divisão não contém exatamente as 2.527 imagens únicas do dataset.")
    if total_by_class != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            "A divisão salva não preserva as quantidades oficiais por classe: "
            f"{total_by_class}. Execute novamente a análise exploratória."
        )
    if splits.get("seed") != SEED:
        raise ValueError(f"A semente da divisão deve ser {SEED}.")

    if "duplicate_groups" not in splits:
        raise ValueError(
            "A divisão foi criada por uma versão antiga que não registra grupos duplicados. "
            "Execute novamente python -m src.prepare_data."
        )
    for group in splits.get("duplicate_groups", []):
        if len(group) < 2:
            raise ValueError(f"Grupo duplicado inválido no splits.json: {group}")
        missing = [path for path in group if path not in path_to_split]
        if missing:
            raise ValueError(f"Imagens de um grupo duplicado não estão na divisão: {missing}")
        assigned_splits = {path_to_split[path] for path in group}
        if len(assigned_splits) != 1:
            raise ValueError(
                f"Vazamento detectado: imagens pixel-idênticas em subconjuntos diferentes: {group}"
            )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)
