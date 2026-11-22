# Extensão opcional — YOLO para classificação

A extensão utiliza YOLO26n-cls no modo `classify`, atribuindo um rótulo à imagem inteira. Não é detecção de objetos e não utiliza caixas delimitadoras.

## Configuração registrada

- Pesos iniciais: `yolo26n-cls.pt` pré-treinado.
- Entrada: 224 × 224; lote 32; até 60 épocas; paciência 10; seed 42.
- Partições: as mesmas 70%/15%/15% do projeto principal.
- Avaliação: métricas por classe, matrizes de confusão, curvas disponíveis, análise de erros e comparação com E3.

## Resultado no teste

O YOLO obteve 90,50% de acurácia e 88,72% de F1 macro nas mesmas 379 imagens de teste utilizadas pelo E3. A comparação é complementar: o YOLO utilizou pesos pré-treinados e transformações próprias do Ultralytics, enquanto E3 começou com pesos aleatórios.

O histórico original preservado contém 28 épocas, mas não registra a acurácia de treino por época. O valor de 99,55% é uma avaliação posterior do checkpoint no conjunto de treino, não uma curva histórica. O script adicional incluído no pacote final registra essa informação em uma nova execução, sem alterar o backup original.
