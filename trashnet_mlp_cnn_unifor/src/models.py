"""Modelos MLP e CNN."""
import tensorflow as tf


def build_mlp(input_shape: tuple[int, int, int], num_classes: int = 6) -> tf.keras.Model:
    "Monta a MLP. Primeiro transforma cada imagem em um vetor de pixels e depois usa camadas densas para aprender a classificar as seis categorias."
    inputs = tf.keras.Input(shape=input_shape, name="imagem_rgb")
    x = tf.keras.layers.Flatten(name="flatten_pixels")(inputs)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_256")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classificacao")(x)
    return tf.keras.Model(inputs, outputs, name="TrashNet_MLP")


def build_cnn(input_shape: tuple[int, int, int], num_classes: int = 6) -> tf.keras.Model:
    "Monta a CNN em blocos de convolução e pooling. As convoluções aprendem padrões visuais, como contornos e texturas, antes da camada que escolhe uma das seis classes."
    inputs = tf.keras.Input(shape=input_shape, name="imagem_rgb")
    x = inputs
    for block, filters in enumerate((32, 64, 128), start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"bloco{block}_conv1")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"bloco{block}_conv2")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"bloco{block}_pool")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.35, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classificacao")(x)
    return tf.keras.Model(inputs, outputs, name="TrashNet_CNN")


def build_model(model_type: str, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    "Escolhe qual arquitetura construir de acordo com a configuração do experimento. Isso mantém o treinamento com uma única chamada tanto para MLP quanto para CNN."
    if model_type.upper() == "MLP":
        return build_mlp(input_shape)
    if model_type.upper() == "CNN":
        return build_cnn(input_shape)
    raise ValueError(f"Tipo de modelo não reconhecido: {model_type}")
