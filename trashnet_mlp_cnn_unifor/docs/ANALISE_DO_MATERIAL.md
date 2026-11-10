# Auditoria do material recebido e alterações realizadas

## 1. Conteúdo do ZIP original

O ZIP continha duas versões de documento (`trabalho_final.docx/.pdf` e `artigo_ava.docx/.pdf`), um diretório de repositório com três scripts Python, dois notebooks, `README.md`, `requirements.txt`, `.gitignore`, um arquivo de instruções de dados e diretórios de resultados vazios. O PDF do artigo para o AVA tinha 9 páginas; o documento extenso tinha 18 páginas.

O material original não continha os 2.527 arquivos de imagem do TrashNet oficial, modelos treinados, históricos completos, checkpoints nem evidências numéricas de que os seis experimentos foram executados. Portanto, não há como confirmar qualquer métrica de desempenho anterior com base somente no ZIP recebido.

## 2. Pontos positivos do trabalho da colega

- O problema de classificação de resíduos foi definido e as seis categorias foram identificadas.
- A proposta dos seis experimentos obrigatórios já estava mapeada.
- Já havia separação conceitual entre análise exploratória, treinamento e avaliação.
- A estratégia de usar pesos de classe e comparar três resoluções estava prevista.
- A CNN usa `GlobalAveragePooling2D`, escolha adequada para conter o crescimento da camada de classificação ao aumentar a resolução.
- O artigo de 9 páginas já seguia uma estrutura próxima à exigida pelo professor.

## 3. Problemas identificados na implementação original

1. **Dependência de caminhos locais:** parte do código esperava uma pasta específica no computador e não tinha uma organização consistente para Colab.
2. **Consumo de RAM:** a abordagem antiga carregava matrizes de imagens na memória, o que se torna caro na resolução 512 × 384.
3. **Compatibilidade com Windows:** o uso do módulo `resource` não é compatível com Windows nativo.
4. **Checagem de arquivos incompleta:** a inspeção anterior verificava uma amostra, não necessariamente todos os arquivos.
5. **Dimensões largura/altura:** havia inversão entre a resolução apresentada na proposta (largura × altura) e a forma esperada por Keras (altura × largura × canais).
6. **Medição de GPU insuficiente:** não existia registro integrado de pico de alocador TensorFlow, VRAM do processo e utilização amostrada.
7. **Retomada limitada:** faltava um mecanismo de recuperação após interrupção do Colab.
8. **Notebooks incompletos:** os notebooks anteriores não constituíam um fluxo claro de instalação, validação de GPU, preparo, treinamento e avaliação.

## 4. Problema encontrado em uma pasta auxiliar

Durante a auditoria havia, na cópia de trabalho do projeto, uma pasta `data/dataset-resized` com apenas 120 imagens (20 por classe), de resolução 64 × 48. Ela não corresponde ao dataset oficial de 2.527 imagens e não é a base permitida para a atividade. Essa pasta de teste foi retirada desta entrega.

O código atualizado exige as quantidades oficiais antes de permitir o treinamento. Assim, uma pasta reduzida ou de teste não poderá ser usada silenciosamente como se fosse o TrashNet oficial.

## 5. Alterações aplicadas nesta versão

- Download automático do ZIP publicado, com conferência SHA-256.
- Extração segura do ZIP e variáveis de ambiente para separar dados locais e resultados persistentes.
- Validação rígida das seis contagens oficiais, do total de 2.527 imagens e da dimensão de referência 512 × 384.
- Inspeção de todos os arquivos: abertura, modo de cor, dimensões, arquivos corrompidos e duplicatas exatas por pixels.
- Uma divisão estratificada 70/15/15 com semente 42, persistida e validada para todos os experimentos.
- Cálculo de pesos somente com a parte de treinamento.
- Pipeline `tf.data` com leitura por lotes, conversão RGB, normalização por 255 e redimensionamento.
- Correção das formas de entrada Keras: `(96,128,3)`, `(192,256,3)` e `(384,512,3)`.
- Retomada do treinamento por época com `BackupAndRestore` e histórico CSV.
- Tentativas de lote 32, 16, 8, 4 e 2 quando ocorre falta de memória.
- Registro de tempo, parâmetros, tamanho do modelo, RAM amostrada, alocador TensorFlow, VRAM do processo (quando disponível) e utilização amostrada da GPU.
- Avaliação dos seis modelos com seleção pelo F1 macro de validação antes de calcular resultados de teste.
- Geração automatizada de tabelas, matrizes de confusão com contagem e percentual, curvas e análise visual de até seis erros.
- Notebook específico para Colab, README atualizado, instruções de GitHub e guia de revisão do artigo.

## 6. O que foi testado nesta estação

- Compilação sintática dos módulos Python: aprovada (`python -m compileall`).
- Teste unitário do preparo de dados: aprovado (`1 passed`).
- Divisão estratificada construída com contagens de referência simuladas: 1.768 treino, 379 validação e 380 teste; validação de classes e ausência de sobreposição aprovada.
- Validação negativa: o código rejeita conjuntos que não contenham as contagens oficiais.

**Limite do teste:** este ambiente de revisão não tem TensorFlow instalado nem GPU TensorFlow disponível. Portanto, não foi possível treinar os seis modelos aqui, medir desempenho nem verificar valores de VRAM reais. O treinamento e as métricas devem ser feitos no Colab/ambiente configurado pela equipe. Os resultados não foram inventados.

## 7. Situação do artigo recebido

O artigo `artigo_ava.pdf` tem 9 páginas, mas não está pronto para entrega. Foram observados:

- Link GitHub fictício/de exemplo (`github.com/equipe/trashnet-mlp-cnn`).
- Tabelas de desempenho/custo sem valores finais.
- Legendas/posições das Figuras 5–8 ainda indicando conteúdo a inserir.
- Trechos da discussão e da conclusão descrevendo tendências esperadas, não resultados observados.
- Necessidade de confirmar que cada afirmação metodológica (integridade de todos os arquivos, dimensões, duplicatas, pesos, batch, arquitetura e tempos) corresponde ao código e à execução final.

O documento extenso de 18 páginas não deve ser enviado como se fosse o artigo de até 10 páginas. Recomenda-se usar `artigo_ava.docx` como base da versão final, atualizando-o somente depois da execução dos experimentos.

## 8. Pendências da equipe

1. Criar o repositório público e ajustar `REPO_URL` no notebook.
2. Executar a inspeção usando o ZIP oficial.
3. Executar E1–E6 e manter os arquivos de modelos/checkpoints no Drive.
4. Executar a avaliação e verificar as três tabelas e as Figuras 1–8.
5. Preencher o artigo com as métricas reais, discutir os erros observados e explicar diferenças de custo.
6. Revisar o texto em conjunto, confirmar referências ABNT, nomes e matrículas e atualizar o link do GitHub.
7. Testar o README/notebook a partir de uma sessão limpa para conferir reprodutibilidade.

A proposta indica prazo de entrega em **27/11/2026 às 23h59**.
