# Estrutura do projeto

| Arquivo | Função |
|---|---|
| `config.py` | Parâmetros comuns e configurações E1–E6. |
| `data_utils.py` | Localização, conferência, deduplicação e divisão dos dados. |
| `prepare_data.py` | Análise inicial e arquivos da preparação. |
| `models.py` | Arquiteturas MLP e CNN. |
| `training_utils.py` | Leitura das imagens e acompanhamento da validação. |
| `train.py` | Treinamento dos experimentos E1–E6. |
| `evaluate.py` | Métricas, matrizes de confusão, curvas e comparações. |
| `train_yolo_bonus.py` | Treinamento e avaliação do YOLO. |
| `train_yolo_bonus_curva.py` | Nova execução YOLO com acurácia de treino por época. |

O notebook `notebooks/00_Execucao_Colab.ipynb` reúne a consulta dos resultados salvos e a execução dos seis experimentos. Os módulos `src/` são compartilhados entre execução local e Colab.
