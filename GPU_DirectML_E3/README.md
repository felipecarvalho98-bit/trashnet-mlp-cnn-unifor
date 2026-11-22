# CNN E3 — AMD DirectML

Implementação independente da CNN E3 em PyTorch/DirectML para a AMD Radeon RX 5500 XT. Não altera os resultados oficiais TensorFlow/Keras.

| Arquivo | Função |
|---|---|
| `PREPARAR_AMBIENTE.bat` | Cria o ambiente isolado e instala as dependências. |
| `CHECK_GPU.bat` | Verifica a GPU e as operações da CNN. |
| `TREINAR_E3_GPU.bat` | Executa o treinamento E3. |
| `FAZER_BACKUP_RESULTADOS.bat` | Compacta os resultados da execução. |
| `src/models.py` | Arquitetura CNN. |
| `src/data_utils.py` | Dataset e partições. |
| `src/train.py` | Treinamento e avaliação. |
| `src/gpu_monitor.py` | Registro de uso e memória da GPU. |

A execução registrada concluiu 60 épocas. No teste, obteve 74,41% de acurácia e 71,78% de F1 macro. O monitoramento registrou 280 amostras, utilização média de 39,05% e pico de 50%.

Este resultado é complementar: a implementação PyTorch/DirectML e o redimensionamento diferem do experimento oficial E3 em TensorFlow/Keras.
