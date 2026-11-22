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
| `src/train_yolo_bonus_curva.py` | Registra também a acurácia de treino por época em uma nova execução YOLO. |
| `notebooks/00_Execucao_Colab.ipynb` | Consulta resultados existentes ou executa novamente E1–E6. |

## Execução

O mesmo código pode ser executado localmente (VS Code/Jupyter) ou no Google Colab. O notebook identifica o ambiente. A opção de consulta usa os resultados salvos; a opção de nova execução grava em uma pasta separada e preserva os arquivos anteriores. Não é necessário montar o Google Drive.

A execução oficial E1–E6 foi realizada em CPU com TensorFlow/Keras. Os resultados estão em `outputs/resultados_finais/`. O YOLO está em `outputs/yolo_bonus/`. A experiência AMD DirectML está separada em `../GPU_DirectML_E3/` e não substitui os experimentos oficiais.

## Resultados e dados

- `docs/RESULTADOS_FINAIS.md`: síntese dos resultados.
- `docs/ESTRUTURA_E_FUNCOES.md`: resumo dos módulos.
- `data/`: dataset usado nos experimentos.
- `outputs/`: partições, métricas, curvas, matrizes e relatórios.

Os checkpoints Keras dos seis modelos oficiais não estavam presentes nos materiais recuperados. As métricas, históricos e demais artefatos disponíveis foram mantidos.
