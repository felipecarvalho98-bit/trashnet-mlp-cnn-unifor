# TrashNet — MLP, CNN e YOLO

Projeto da atividade de Redes Neurais Artificiais (UNIFOR). Reúne os seis experimentos obrigatórios E1–E6, a extensão YOLO e os resultados consolidados.

## Código

| Arquivo | O que faz |
|---|---|
| `src/prepare_data.py` | Analisa o TrashNet, confere duplicatas e prepara as partições. |
| `src/models.py` | Define as arquiteturas MLP e CNN. |
| `src/train.py` | Executa os treinamentos E1–E6. |
| `src/evaluate.py` | Calcula métricas, matrizes de confusão e comparações. |
| `src/train_yolo_bonus.py` | Treina e avalia a extensão YOLO. |
| `notebooks/00_Execucao_Colab.ipynb` | Consulta resultados existentes ou executa novamente E1–E6. |

## Execução

O mesmo código pode ser executado localmente (VS Code/Jupyter) ou no Google Colab. O notebook identifica o ambiente. A opção de consulta usa os resultados salvos; a opção de nova execução grava em uma pasta separada e preserva os arquivos anteriores. Não é necessário montar o Google Drive.

A execução oficial E1–E6 foi realizada em CPU com TensorFlow/Keras. Os resultados estão em `outputs/resultados_finais/`. O YOLO é uma extensão complementar, com resultados em `outputs/yolo_bonus/`.

## Síntese dos resultados

E3 foi selecionado pela macro-F1 de validação (76,31%) e obteve 76,78% de acurácia e 74,76% de F1 macro no teste. O YOLO26n-cls alcançou 90,50% de acurácia e 88,72% de F1 macro, utilizando pesos pré-treinados. A comparação é complementar, pois as configurações de inicialização e entrada são diferentes.

A experiência AMD DirectML da CNN E3 está documentada separadamente no pacote final de entrega; não substitui os seis resultados oficiais.
