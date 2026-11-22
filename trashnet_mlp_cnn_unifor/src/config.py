"""Configurações dos experimentos TrashNet."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TRASHNET_DATA_DIR", ROOT_DIR / "data")).expanduser()
OUTPUT_DIR = Path(os.environ.get("TRASHNET_OUTPUT_DIR", ROOT_DIR / "outputs")).expanduser()
FINAL_DIR = OUTPUT_DIR / "resultados_finais"

# A ordem é fixa porque ela também define os índices usados nas saídas da rede.
CLASS_NAMES = ("glass", "paper", "cardboard", "plastic", "metal", "trash")
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Parâmetros iguais entre os experimentos para que a comparação seja mais justa.
SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 16
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3

# O enunciado informa largura x altura; aqui guardamos altura x largura porque é a ordem do TensorFlow.
EXPERIMENTS = {
    "E1": {
        "name": "MLP_128x96", "model_type": "MLP", "image_size": (96, 128),
        "class_weights": False, "description": "MLP 128x96 sem pesos de classe",
    },
    "E2": {
        "name": "MLP_PESOS_128x96", "model_type": "MLP", "image_size": (96, 128),
        "class_weights": True, "description": "MLP 128x96 com pesos de classe",
    },
    "E3": {
        "name": "CNN_128x96", "model_type": "CNN", "image_size": (96, 128),
        "class_weights": False, "description": "CNN 128x96 sem pesos de classe",
    },
    "E4": {
        "name": "CNN_PESOS_128x96", "model_type": "CNN", "image_size": (96, 128),
        "class_weights": True, "description": "CNN 128x96 com pesos de classe",
    },
    "E5": {
        "name": "CNN_256x192", "model_type": "CNN", "image_size": (192, 256),
        "class_weights": False, "description": "CNN 256x192 sem pesos de classe",
    },
    "E6": {
        "name": "CNN_512x384", "model_type": "CNN", "image_size": (384, 512),
        "class_weights": False, "description": "CNN 512x384 sem pesos de classe",
    },
}


def experiment_dir(experiment_id: str) -> Path:
    "Monta o caminho onde ficam os arquivos de um experimento. Assim, cada teste E1–E6 mantém seus modelos, histórico e métricas separados."
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Experimento desconhecido: {experiment_id}. Opções: {list(EXPERIMENTS)}")
    return OUTPUT_DIR / f"{experiment_id}_{EXPERIMENTS[experiment_id]['name']}"
