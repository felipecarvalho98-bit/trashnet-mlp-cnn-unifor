from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATASET_ZIP = DATA_DIR / "dataset-resized_TrashNet_mesmo_dataset.zip"
EXTRACT_DIR = DATA_DIR / "dataset_extraido"
SPLITS_DIR = OUTPUT_DIR / "preparacao"

CLASSES = ("glass", "paper", "cardboard", "plastic", "metal", "trash")
SEED = 42
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 96
BATCH_SIZE = 16
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 1e-3
REDUCE_LR_PATIENCE = 4
MIN_LEARNING_RATE = 1e-5
SAMPLE_GPU_EVERY_SECONDS = 2.0

# Referência do experimento E3 oficial executado em CPU (para comparação posterior).
CPU_BASELINE = {
    "experiment": "E3 TensorFlow/Keras CPU",
    "test_accuracy": 0.767810,
    "test_macro_f1": 0.7476133,
    "note": (
        "Comparação indicativa: a arquitetura e a divisão foram mantidas, mas a implementação "
        "PyTorch/DirectML e o redimensionamento podem produzir pequenas diferenças em relação ao TensorFlow/Keras."
    ),
}
