"""Verificação das bibliotecas e dispositivos disponíveis."""
import sys
import platform

print("Python:", sys.version.split()[0])
print("Executável:", sys.executable)
print("Sistema:", platform.platform())

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("Dispositivos:", tf.config.list_physical_devices())
except Exception as exc:
    print("ERRO ao importar TensorFlow:", repr(exc))
    raise

import numpy, pandas, sklearn, PIL, matplotlib, psutil, openpyxl
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("Pillow:", PIL.__version__)
print("Matplotlib:", matplotlib.__version__)
print("psutil:", psutil.__version__)
print("openpyxl:", openpyxl.__version__)
print("\nAmbiente verificado com sucesso.")
