# Passo a passo — Google Colab, VS Code e GitHub

## Parte 1 — Criar o repositório no GitHub

1. Entre na sua conta em https://github.com/.
2. Clique em **New repository**.
3. Nome sugerido: `trashnet-mlp-cnn-unifor`.
4. Descrição sugerida: `Classificação de resíduos do TrashNet com MLP e CNN — atividade de Redes Neurais Artificiais`.
5. Marque **Public**, para o professor conseguir acessar pelo link.
6. **Não** marque para criar README, `.gitignore` ou licença, porque o projeto já tem esses arquivos.
7. Clique em **Create repository**.
8. Copie a URL HTTPS exibida. Ela terá o formato `https://github.com/SEU_USUARIO/trashnet-mlp-cnn-unifor.git`.

### Enviar a pasta do projeto pelo VS Code/terminal

Baixe e extraia o ZIP revisado, abra a pasta raiz `trashnet_mlp_cnn_unifor` no VS Code e execute no terminal:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Estrutura inicial do projeto TrashNet"
git remote add origin https://github.com/SEU_USUARIO/trashnet-mlp-cnn-unifor.git
git push -u origin main
```

Troque `SEU_USUARIO` pelo seu nome de usuário e use a URL real do repositório recém-criado. Se o Git solicitar autenticação, conclua pelo navegador/Git Credential Manager; não coloque senha ou token diretamente no código.

Depois do primeiro envio, as atualizações são:

```bash
git add README.md docs notebooks src tests outputs/splits.json outputs/metricas outputs/figuras
git status
git commit -m "Atualiza experimento e resultados"
git push
```

Antes de cada commit, confira `git status`. Não envie ZIP do dataset, pasta de imagens, modelos `.keras`, checkpoints, tokens ou credenciais. O `.gitignore` já exclui esses arquivos.

## Parte 2 — Preparar o notebook no Google Colab

1. Abra `notebooks/00_Execucao_Colab.ipynb` no Colab.
2. No menu **Ambiente de execução → Alterar o tipo de ambiente de execução**, selecione **GPU** e salve.
3. Na primeira célula, altere `REPO_URL` para a URL HTTPS real do seu repositório.
4. Execute a célula de montagem do Google Drive e clone o repositório dentro da pasta `MyDrive/IA_UNIFOR/`.
5. Execute as células de dependências e GPU. Confirme que `GPUs:` mostra pelo menos um dispositivo e que a multiplicação de matrizes informa `/device:GPU:0`. Se não aparecer, volte à configuração do runtime, selecione GPU e reinicie.
6. Execute a célula de análise exploratória. O código baixa o arquivo oficial, confere o hash e exige as 2.527 imagens. Confira as contagens e as Figuras 1–4.
7. Execute primeiro E3 como teste de fluxo completo. Ele pode levar tempo; não é um teste de uma única época.
8. Quando E3 terminar e a configuração estiver registrada, execute `python -m src.train --all --skip-existing` para concluir os modelos restantes sem refazer E3.
9. Se o Colab desconectar, reabra o mesmo notebook, monte o Drive, use o mesmo caminho de projeto e execute novamente a célula de treinamento. Os checkpoints por época ficam no Drive. Não apague `outputs/checkpoints/` antes de concluir o treino.
10. Só depois de existirem os seis `.keras` e os seis arquivos de configuração, execute a avaliação.
11. Baixe/revise as três tabelas e as Figuras 5–8. Salve no repositório os arquivos pequenos gerados. Mantenha modelos e checkpoints no Drive.

### Organização dos arquivos no Colab

- Código, resultados, históricos e checkpoints: pasta de projeto no Drive.
- ZIP e imagens: `/content/trashnet_data`, disco local da VM para acelerar leitura.
- Modelos `.keras`: `outputs/modelos/`, no Drive e ignorados pelo Git.

Se o runtime for totalmente recriado, a pasta `/content/trashnet_data` é temporária e o dataset precisa ser baixado/extráido novamente. Os resultados gravados no Drive permanecem.

## Parte 3 — Execução local pelo VS Code (Windows)

Use Python 3.11. Abra PowerShell na pasta raiz do projeto:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, use o terminal `cmd` e execute:

```bat
.venv\Scripts\activate.bat
```

Comandos:

```powershell
python -m src.prepare_data
python -m src.train --exp E3
python -m src.train --all --skip-existing
python -m src.evaluate
```

No Windows nativo, a execução TensorFlow é em CPU. Para GPU NVIDIA, use Colab ou WSL2 configurado com o driver apropriado. Não compare tempos de máquina local e Colab como se o hardware fosse o mesmo.

## Parte 4 — Compartilhar modelos com o professor

A proposta permite que modelos grandes sejam armazenados no Google Drive. Após os seis treinos:

1. Abra `outputs/modelos/` e confirme os seis arquivos `modelo_E1.keras` até `modelo_E6.keras`.
2. Compartilhe a pasta `outputs/modelos/` (e, se necessário, `outputs/historicos/` e `outputs/metricas/`) com a opção de acesso adequada exigida pelo professor, normalmente **Qualquer pessoa com o link — Leitor**.
3. Copie o link e inclua no README, identificando quais arquivos estão na pasta.
4. Teste o link em janela anônima ou conta sem permissão de edição para confirmar que o professor consegue abrir.

Não suba arquivos de modelo e checkpoints no GitHub: podem exceder limites de tamanho e o projeto já usa Drive para eles.

## Parte 5 — Atualizar o artigo após a execução

Use `docs/ROTEIRO_REVISAO_ARTIGO.md`. Não inclua números antes de conferir os CSVs/JSONs; não mantenha legendas de figuras ainda inexistentes; e não escreva a conclusão com base no que se esperava que acontecesse. A equipe deve interpretar as próprias evidências e confirmar referências, nomes completos e matrículas.

## Checklist final

- [ ] Repositório público criado e link real no README/notebook/artigo.
- [ ] ZIP oficial verificado por hash e 2.527 imagens validadas.
- [ ] Divisão 70/15/15, seed 42 e sem sobreposição.
- [ ] E1, E2, E3, E4, E5 e E6 treinados e salvos.
- [ ] Hardware, lotes, parâmetros, tempos e memória registrados.
- [ ] Três tabelas CSV completas e conferidas.
- [ ] Figuras 1–8 presentes e legíveis.
- [ ] Link de Drive dos modelos testado pelo modo leitor.
- [ ] Artigo atualizado até 10 páginas com discussão e conclusão baseadas em resultados reais.
- [ ] Todas as pessoas do grupo conseguem explicar os principais passos e decisões.

**Prazo indicado na proposta:** 27/11/2026 às 23h59.
