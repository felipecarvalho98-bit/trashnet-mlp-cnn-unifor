"""Smoke test sem TensorFlow: valida inspeção, divisão e integridade dos splits."""
from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.common import (
    CLASSES, EXPECTED_CLASS_COUNTS, validate_reference_dataset_counts, validate_splits,
)
from src.prepare_data import inspect_dataset, make_stratified_splits


def test_inspection_and_splits():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for class_index, class_name in enumerate(CLASSES):
            folder = root / class_name
            folder.mkdir()
            for image_index in range(20):
                # Padrões diferentes evitam duplicatas de pixels entre arquivos.
                pixels = np.zeros((24, 32, 3), dtype=np.uint8)
                pixels[:, :, class_index % 3] = (class_index * 31 + image_index * 7) % 255
                pixels[image_index % 24, :, :] = 255
                Image.fromarray(pixels, mode="RGB").save(folder / f"img_{image_index:03d}.jpg")

        records, info = inspect_dataset(root)
        assert len(records) == 120
        assert not info["corrupted_files"]
        assert not info["exact_pixel_duplicates"]

        try:
            validate_reference_dataset_counts(root)
        except ValueError as exc:
            assert "2.527 imagens" in str(exc)
        else:
            raise AssertionError("Dataset de teste pequeno não deveria ser aceito como TrashNet oficial.")

        splits = make_stratified_splits(records)
        assert splits["totals_by_split"] == {"train": 84, "val": 18, "test": 18}
        paths = splits["paths"]
        assert not (set(paths["train"]) & set(paths["val"]))
        assert not (set(paths["train"]) & set(paths["test"]))
        assert not (set(paths["val"]) & set(paths["test"]))
        for split in ("train", "val", "test"):
            assert all(splits["counts_by_split_and_class"][split][name] > 0 for name in CLASSES)


if __name__ == "__main__":
    test_inspection_and_splits()
    print("OK: smoke test de inspeção e divisão concluído.")


def test_official_split_sizes_and_validation():
    records = []
    for label, class_name in enumerate(CLASSES):
        for image_index in range(EXPECTED_CLASS_COUNTS[class_name]):
            # Simula dois grupos de duplicatas exatas, inclusive um entre classes,
            # para testar que o grupo não é separado entre treino/validação/teste.
            if (label, image_index) in {(0, 0), (1, 0)}:
                pixel_hash = "duplicata_entre_classes"
            elif (label, image_index) in {(2, 0), (2, 1)}:
                pixel_hash = "duplicata_mesma_classe"
            else:
                pixel_hash = f"unico_{label}_{image_index}"
            records.append({
                "path": f"{class_name}/{class_name}_{image_index:04d}.jpg",
                "label": label,
                "pixel_sha256": pixel_hash,
            })

    splits = make_stratified_splits(records)
    validate_splits(splits)
    assert splits["totals_by_split"] == {"train": 1768, "val": 379, "test": 380}
    assert len(set(splits["paths"]["train"]) | set(splits["paths"]["val"]) | set(splits["paths"]["test"])) == 2527
    path_to_split = {path: split for split, paths in splits["paths"].items() for path in paths}
    assert len(splits["duplicate_groups"]) == 2
    for group in splits["duplicate_groups"]:
        assert len({path_to_split[path] for path in group}) == 1
