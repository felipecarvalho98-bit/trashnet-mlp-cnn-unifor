# Classificação de resíduos com MLP e CNN — TrashNet

Projeto da atividade final de **Redes Neurais Artificiais (UNIFOR)**. O objetivo é comparar MLP e CNN, estudar o efeito de pesos de classe e medir o custo de treinar CNNs em três resoluções.

> **Situação deste repositório:** o código e o notebook estão organizados para execução. Os resultados numéricos só existirão depois que a equipe executar os seis treinamentos e a avaliação. Não há métricas fictícias incluídas.

## Requisitos da atividade

| Experimento | Arquitetura | Entrada (largura × altura) | Pesos de classe |
|---|---|---:|---|
| E1 | MLP | 128 × 96 | Não |
| E2 | MLP | 128 × 96 | Sim |
| E3 | CNN | 128 × 96 | Não |
| E4 | CNN | 128 × 96 | Sim |
| E5 | CNN | 256 × 192 | Não |
| E6 | CNN | 512 × 384 | Não |

As classes são mantidas na ordem definida pelo professor: `glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`.

## Estrutura

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── trashnet_mlp_cnn_unifor/\n    ├── src/
│   ├── common.py          # caminhos, download, integridade e validações
│   ├── prepare_data.py    # inspeção, divisão, pesos e Figuras 1–4
│   ├── models.py          # arquiteturas MLP e CNN
│   ├── train.py           # treinamento E1–E6 e medição de custo
│   └── evaluate.py        # seleção por validação, teste, tabelas e Figuras 5–8
    ├── notebooks/
│   └── 00_Execucao_Colab.ipynb
    ├── data/
│   └── LEIA-ME.txt        # instruções; o dataset não é versionado
    ├── outputs/
│   ├── splits.json        # criado após a inspeção
│   ├── metricas/          # inspeção, métricas e tabelas CSV
│   ├── historicos/        # histórico por época
│   ├── modelos/           # arquivos .keras (fora do Git)
│   ├── checkpoints/       # retomada de sessões interrompidas (fora do Git)
│   └── figuras/           # Figuras 1–8
    ├── tests/
│   └── test_prepare_data.py
    └── docs/
    ├── ANALISE_DO_MATERIAL.md
    ├── ANALISE_GPU.md
    ├── PASSO_A_PASSO_COLAB_VSCODE_GITHUB.md
    └── ROTEIRO_REVISAO_ARTIGO.md
```

## 1. Dataset

O arquivo oficial `data/dataset-resized.zip` está versionado neste repositório. O ZIP está versionado em `trashnet_mlp_cnn_unifor/data/dataset-resized.zip`. O notebook clona o repositório no início e utiliza esse arquivo diretamente, sem baixar o dataset de outra fonte. O código confere o SHA-256 antes de extrair.

O programa exige as contagens oficiais da proposta: glass 501, paper 594, cardboard 403, plastic 482, metal 410 e trash 137 (2.527 imagens no total). Essa validação é intencional: impede que uma pasta pequena de teste seja confundida com o dataset oficial. A inspeção abre todos os arquivos, confere dimensões, modo de cor, imagens corrompidas e duplicatas exatas por pixels. A execução para se encontrar algo fora do esperado, em vez de treinar silenciosamente com dados incorretos.

## 2. Colab — caminho recomendado para GPU

1. Crie um repositório público conforme `docs/PASSO_A_PASSO_COLAB_VSCODE_GITHUB.md`.
2. Abra `notebooks/00_Execucao_Colab.ipynb` no Google Colab.
3. Em **Ambiente de execução → Alterar o tipo de ambiente de execução**, escolha GPU.
4. A primeira célula clona/atualiza automaticamente o repositório público correto; não precisa editar caminhos nem montar o Google Drive.
5. Execute as células na ordem. O código, o ZIP e a pasta `outputs/` ficam em `/content` na VM do Colab. Os resultados são temporários e devem ser baixados em backup antes de encerrar a sessão.
6. Execute a análise exploratória, os seis treinamentos E1–E6 e, ao final, rode a avaliação.

O Colab não garante um tipo específico de GPU nem disponibilidade contínua. O notebook confere se o TensorFlow detectou uma GPU e executa uma multiplicação de matrizes nela. Consulte `docs/ANALISE_GPU.md` para entender as medições e as limitações.

## 3. VS Code no Windows

Use Python 3.11 em um ambiente virtual. No Windows nativo, o TensorFlow atual funciona em CPU; para GPU NVIDIA, prefira Colab ou WSL2 com o driver e a configuração de CUDA compatíveis.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Na pasta raiz do repositório:

```powershell
python -m src.prepare_data
python -m src.train --exp E3
python -m src.train --all --skip-existing
python -m src.evaluate
```

`E3` é um primeiro treino completo para testar o ambiente, não um teste de uma única época. Depois, `--all --skip-existing` executa os experimentos restantes sem repetir os que já têm modelo e configuração salvos.

## 4. Comandos principais

```bash
# Inspeciona o conjunto, cria a divisão e gera as Figuras 1–4
python -m src.prepare_data

# Treina um experimento completo
python -m src.train --exp E3

# Treina E1 até E6 e pula o que já foi concluído
python -m src.train --all --skip-existing

# Calcula métricas e gera tabelas e Figuras 5–8
python -m src.evaluate
```

A divisão é estratificada (70%/15%/15%), usa semente 42 e é gravada em `outputs/splits.json`. A validação impede sobreposição entre treino, validação e teste. Os pesos de classe são calculados apenas com os rótulos de treino. O conjunto de teste só é usado depois da seleção do modelo pelo F1 macro de validação.

## 5. Dimensões das imagens

A proposta informa largura × altura. O TensorFlow/Keras recebe altura × largura × canais:

| Resolução da proposta | Forma de entrada Keras |
|---:|---:|
| 128 × 96 | `(96, 128, 3)` |
| 256 × 192 | `(192, 256, 3)` |
| 512 × 384 | `(384, 512, 3)` |

As imagens são convertidas para RGB e os pixels são normalizados dividindo por 255. A CNN usa três blocos convolucionais e `GlobalAveragePooling2D`, reduzindo o crescimento de parâmetros na classificação para as resoluções maiores. A MLP recebe os pixels achatados em vetor.

## 6. Treinamento, memória e retomada

- O pipeline usa `tf.data` para ler, decodificar e redimensionar imagens em lotes, evitando manter todas as imagens em arrays grandes de RAM.
- O lote tenta 32, 16, 8, 4 e 2, nessa ordem. Se ocorrer erro de falta de memória, a tentativa recomeça com lote menor e a configuração usada fica registrada.
- `EarlyStopping` monitora `val_loss`, com paciência 8 e restauração dos melhores pesos; máximo de 60 épocas.
- `BackupAndRestore` grava checkpoints por época durante a sessão atual. Como o notebook não monta o Google Drive, os checkpoints e demais resultados ficam na VM temporária do Colab; baixe o ZIP de backup antes que a sessão seja encerrada. O histórico CSV permite reconstruir o tempo acumulado das épocas concluídas.
- A medição de RAM é uma amostragem de RSS do processo. O TensorFlow informa o pico do seu alocador de GPU; `nvidia-smi`, quando disponível, informa a memória do processo e utilização amostrada do dispositivo. A utilização é do dispositivo como um todo, não atribuível apenas ao processo da equipe.
- O tempo médio e total reportados somam os tempos medidos por época. O cronômetro de `model.fit` inclui o carregamento/decodificação/redimensionamento durante o treinamento e exclui download, inspeção prévia, montagem dos datasets, compilação e gravação do modelo.

Consulte `docs/ANALISE_GPU.md` antes de interpretar esses números.

## 7. Resultados gerados

**Após análise exploratória:** `outputs/metricas/inspecao_dataset.json`, `outputs/splits.json` e Figuras 1–4.

**Após os seis treinamentos:** seis arquivos `.keras`, seis históricos JSON, seis configurações JSON e CSVs de época em `outputs/historicos/` e `outputs/metricas/`.

**Após avaliação:**

- `outputs/metricas/tabela5_custo.csv`
- `outputs/metricas/tabela6_desempenho.csv`
- `outputs/metricas/tabela7_por_classe.csv`
- `outputs/metricas/resultados_completos.json`
- Figuras 5–8 em `outputs/figuras/`

As matrizes de confusão exibem contagens e porcentagens por classe real. As métricas por classe usam `zero_division=0`. O modelo de referência é escolhido pelo maior F1 macro de validação; somente depois os modelos são comparados no teste.

## 8. GitHub e arquivos grandes

Publique no GitHub o código, notebook, documentação, divisão, métricas, tabelas e figuras. O `.gitignore` exclui dataset, modelos `.keras`, checkpoints e ambientes virtuais. Os modelos e checkpoints não são enviados ao GitHub. Como o notebook executa sem Drive, baixe o ZIP de backup gerado no Colab e guarde-o no computador. Depois compartilhe os arquivos com o professor pelo meio de armazenamento escolhido pela equipe.

Os arquivos recebidos da colega e a proposta do professor foram preservados sem alteração em `documentos_originais/`; a revisão do conteúdo do artigo está documentada em `docs/ANALISE_DO_MATERIAL.md` e `docs/ROTEIRO_REVISAO_ARTIGO.md`.

## 9. Artigo e autoria acadêmica

O artigo recebido deve ser atualizado depois que os seis modelos forem executados. Não preencham tabelas com números de exemplo nem mantenham conclusões formuladas como expectativa. A equipe deve interpretar os próprios resultados, registrar decisões realmente tomadas, explicar limitações e revisar as referências. A proposta do professor permite IA como apoio à geração de código, mas exige que os integrantes compreendam, revisem e conduzam o trabalho. Veja `docs/ROTEIRO_REVISAO_ARTIGO.md`.

## Fontes

- Dataset TrashNet: https://huggingface.co/datasets/garythung/trashnet
- Repositório original: https://github.com/garythung/trashnet
- TensorFlow — uso de GPU: https://www.tensorflow.org/guide/gpu
- TensorFlow — instalação e plataformas: https://www.tensorflow.org/install/pip
- Google Colab — FAQ: https://research.google.com/colaboratory/faq.html
