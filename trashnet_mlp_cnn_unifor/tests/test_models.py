"""Teste básico da estrutura dos modelos (requer TensorFlow instalado)."""
import pytest

tf = pytest.importorskip("tensorflow")
from src.models import build_model


def test_mlp_output_has_six_classes():
    model = build_model("MLP", (96, 128, 3))
    assert model.output_shape == (None, 6)


def test_cnn_output_has_six_classes():
    model = build_model("CNN", (96, 128, 3))
    assert model.output_shape == (None, 6)
