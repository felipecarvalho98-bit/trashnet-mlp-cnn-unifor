# Objetivo extra — YOLO para classificação

Garantir o ponto bônus. A extensão consiste em ajustar um classificador YOLO pré-treinado no modo `classify`; não é detecção de objetos e não utiliza caixas delimitadoras.

## Procedimento experimental

1. Manter as seis classes: `glass`, `paper`, `cardboard`, `plastic`, `metal` e `trash`.
2. Reutilizar a divisão original 70%/15%/15% de `splits.csv`;
3. Disponibilizar `splits.csv`, `dataset_metadata.json` e o mesmo ZIP usado em E1–E6. O script verifica as quantidades e o SHA-256 de cada imagem.
4. Converter a divisão para as pastas `train/`, `val/` e `test/` esperadas pelo Ultralytics.
5. Carregar os pesos pré-treinados de classificação `yolo26n-cls.pt` e realizar fine-tuning no TrashNet.
6. Registrar variante, versão, origem dos pesos, tamanho de entrada, transformações, lote, épocas, paciência, hardware e tempo. A configuração inicial é 224 px, batch 32, até 60 épocas, patience 10 e seed 42.
7. Acompanhar as métricas de validação e manter o checkpoint selecionado pelo critério interno do Ultralytics (top-1 de validação). O script calcula também F1 macro na validação. O teste só é avaliado depois da seleção.
8. Comparar as métricas de teste e o custo computacional com a CNN E3, selecionada pelo maior F1 macro de validação.
9. Apresentar configuração, curvas, matrizes, tabela comparativa e análise dos erros no mesmo artigo, dentro do limite de 10 páginas.

## Evidências e pontuação

| Item | Valor | Evidência registrada |
|---|---:|---|
| Configuração e treinamento | 0,25 | `resumo_yolo.json`, código `src/train_yolo_bonus.py` e checkpoint `best.pt`. |
| Avaliação | 0,25 | Acurácia, F1 macro, precisão, recall, F1 e suporte por classe; matriz de confusão; curvas de perda e acurácia de treino/validação. |
| Comparação | 0,25 | `comparacao_yolo_E3.csv` e `comparacao_yolo_E3.png`, com métricas de teste e custo. |
| Discussão | 0,25 | Erros, desempenho, custo, pré-treinamento, resolução e transformações. |

## Artefatos

`outputs/yolo_bonus/` contém resumo JSON, histórico CSV, métricas por classe, previsões, análise dos erros, matrizes de confusão em contagens e porcentagens, curvas de aprendizagem e comparação com E3. O dataset convertido e a pasta temporária de execução são excluídos do controle de versão. O checkpoint `best.pt` pode ser disponibilizado por link público quando exceder o limite prático do repositório.

A acurácia de treino é calculada ao final de cada época sobre o conjunto de treino, utilizando o pipeline de transformações de treinamento. Essa passagem adicional é contabilizada no tempo total do YOLO. A acurácia de validação é registrada pelo Ultralytics.

## Transformações e interpretação

O modo de classificação do Ultralytics utiliza recorte aleatório no treinamento e recorte central na validação/inferência por padrão. Como as imagens TrashNet têm proporção 4:3 e a entrada padrão é quadrada, parte da imagem pode ser recortada. Essa diferença em relação ao redimensionamento usado pela CNN deve ser considerada na discussão. A documentação oficial descreve como substituir esses recortes por letterbox caso a equipe decida preservar todo o enquadramento.

O notebook exige `splits.csv` e `dataset_metadata.json` da preparação original quando o modo resultados existentes é selecionado. Também verifica 2.524 imagens, partições com 1.766/379/379 amostras e SHA-256 de cada arquivo antes de copiar as imagens. O ZIP do dataset precisa ser o mesmo usado em E1–E6; arquivos diferentes são recusados para evitar uma comparação com imagens distintas.

O bônus não substitui E1–E6 e não exige que o YOLO supere a E3.

## Documentação oficial

- Classificação YOLO: https://docs.ultralytics.com/tasks/classify
- Estrutura de datasets de classificação: https://docs.ultralytics.com/datasets/classify
- Treinamento: https://docs.ultralytics.com/modes/train
- Callbacks: https://docs.ultralytics.com/usage/callbacks

Entrega prevista até dia 27!