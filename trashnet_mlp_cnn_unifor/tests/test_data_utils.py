from pathlib import Path
import pandas as pd
import pytest

from src.data_utils import create_splits
from src.config import CLASS_NAMES


def make_frame(n_per_class=40):
    rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            rows.append({
                "path": f"{class_name}/{i}.jpg",
                "relative_path": f"{class_name}/{i}.jpg",
                "class_name": class_name,
                "class_index": class_index,
                "width": 10,
                "height": 10,
                "sha256": f"{class_name}-{i}",
            })
    return pd.DataFrame(rows)


def test_create_splits_has_three_partitions_and_no_hash_leakage():
    result = create_splits(make_frame())
    assert set(result.split) == {"train", "validation", "test"}
    assert result.sha256.is_unique
    assert result.groupby("sha256").split.nunique().max() == 1
    assert len(result) == 240


def test_create_splits_is_reproducible():
    frame = make_frame()
    a = create_splits(frame)
    b = create_splits(frame)
    assert a[["relative_path", "split"]].equals(b[["relative_path", "split"]])
