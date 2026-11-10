# Roteiro para revisar o artigo técnico

O artigo recebido tem uma base aproveitável, mas não deve ser tratado como versão final até existirem resultados dos seis experimentos. O objetivo desta revisão é fortalecer autoria, clareza e rigor científico por meio das decisões reais da equipe — não mascarar o uso de ferramentas de IA nem tentar contornar detectores.

## O que corrigir no artigo atual

1. Trocar o endereço GitHub fictício pelo repositório público real.
2. Completar as tabelas de desempenho/custo somente com os CSVs de saída.
3. Substituir as posições/legendas pendentes das Figuras 5–8 pelos arquivos efetivamente gerados.
4. Remover frases de expectativa como “espera-se que a CNN...” quando a seção apresenta resultados. Trocar por observações apoiadas nas curvas e tabelas.
5. Conferir todas as afirmações metodológicas contra o código executado: número de imagens verificadas, dimensões, duplicatas, normalização, forma de entrada, pesos, lote, épocas, early stopping e hardware.
6. Explicar quando os lotes diferirem entre resoluções por falta de memória.
7. Distinguir o resultado publicado no artigo de Yang e Thung do valor posterior informado no README do repositório TrashNet, conforme orientação do professor.
8. Conferir citações e referências no padrão ABNT e o limite de 10 páginas.

## Proposta de organização (até 10 páginas)

### 1. Introdução

Apresentar o problema de classificação de resíduos, justificar o interesse em reconhecer materiais por imagem e declarar o objetivo: comparar MLP e CNN, avaliar pesos de classe e investigar desempenho/custo em três resoluções.

### 2. Materiais e métodos

Descrever: origem do TrashNet; 2.527 imagens e seis classes; verificação de integridade; conversão RGB e normalização; divisão estratificada 70/15/15 com seed 42; arquiteturas; configuração do otimizador/perda/early stopping; cálculo dos pesos somente com treino; hardware, software e política de lote.

### 3. Resultados

Inserir tabela de desempenho dos seis experimentos, tabela de custo, tabela por classe, curvas de aprendizagem, seis matrizes de confusão e análise de até seis erros do modelo selecionado pela validação. Identificar cada figura e tabela e explicar o que ela mostra.

### 4. Discussão

Responder com evidências: MLP versus CNN em 128 × 96; impacto dos pesos na classe `trash` e nas demais; impacto da resolução em F1 macro/acurácia e tempo/memória; pares de classes confundidos; características visuais que podem justificar os erros; limitações do conjunto e diferença metodológica para o estudo de referência.

### 5. Conclusão

Resumir os achados efetivamente observados, indicar qual configuração teve melhor F1 macro de validação e explicar se o ganho de resolução compensou o custo. Não declarar um vencedor antes da avaliação.

### 6. Referências

Listar dataset, artigo de referência, documentação das bibliotecas e outras fontes realmente utilizadas.

## Como deixar o texto autenticamente da equipe

- Cada integrante deve explicar oralmente o código, o split, as arquiteturas, as métricas e as limitações.
- Registrem no caderno do grupo por que escolheram a configuração, que dificuldade apareceu e como foi resolvida.
- Na discussão, citem exemplos concretos das matrizes de confusão e das imagens classificadas incorretamente.
- Não usem afirmações genéricas sem uma figura, tabela, fonte ou observação de execução que as sustente.
- Escrevam a primeira versão da interpretação a partir dos resultados e revisem linguagem, coerência e referências em conjunto.
- Declarem ferramentas de apoio de acordo com a política da disciplina, sem esconder ou inventar autoria.

## Campos que só podem ser preenchidos depois dos treinos

| Informação | Arquivo de origem |
|---|---|
| Melhor experimento pelo F1 macro de validação | `outputs/metricas/modelo_referencia_erros.json` |
| Acurácia e F1 macro no teste | `outputs/metricas/tabela6_desempenho.csv` |
| Precisão, recall, F1 e suporte por classe | `outputs/metricas/tabela7_por_classe.csv` |
| Parâmetros, tamanho, tempo e memória | `outputs/metricas/tabela5_custo.csv` |
| Curvas por época | `outputs/historicos/historico_E*.json` e Figura 5 |
| Erros e probabilidade prevista | `outputs/metricas/modelo_referencia_erros.json` e Figura 8 |

Não preencher os campos acima com valores estimados, exemplos de internet ou resultados de outra equipe.
