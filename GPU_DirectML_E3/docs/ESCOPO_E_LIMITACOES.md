# Escopo e limitações conhecidas

1. Este primeiro teste é somente o E3 (CNN 128 × 96, sem pesos de classe). A escolha evita refazer os seis experimentos antes de confirmar se a arquitetura original é compatível com DirectML.
2. A inicialização é aleatória e o treinamento começa do zero. Nenhum peso pré-treinado é carregado.
3. O projeto principal usa TensorFlow/Keras. Esta cópia usa PyTorch, pois a instalação DirectML já foi testada separadamente na máquina. Não existe troca automática de backend no projeto original.
4. O redimensionamento é feito com Pillow em RGB e interpolação bilinear, mantendo a entrada 128 × 96 e a normalização 0–1. Podem existir diferenças numéricas pequenas em relação ao redimensionamento do TensorFlow.
5. A amostragem dos contadores do Windows ocorre aproximadamente a cada dois segundos; picos curtos podem passar entre amostras.
6. O sucesso do teste rápido comprova que a CNN E3 completa consegue executar forward, loss, backward e Adam no dispositivo detectado. O treinamento integral só será considerado validado depois de terminar e gerar métricas.
7. O pacote não modifica arquivos, resultados ou o repositório do projeto original.
