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

## Parte 2 — Preparar o notebook no Google Colab (sem Google Drive)

1. Abra o notebook atualizado: [00_Execucao_Colab.ipynb](https://colab.research.google.com/github/felipecarvalho98-bit/trashnet-mlp-cnn-unifor/blob/main/trashnet_mlp_cnn_unifor/notebooks/00_Execucao_Colab.ipynb).
2. Em **Ambiente de execução → Alterar o tipo de ambiente de execução**, selecione **GPU** e salve. A GPU disponível depende da cota e dos recursos do Colab.
3. Execute todas as células na ordem, de cima para baixo. A primeira célula clona o repositório público para `/content/trashnet-mlp-cnn-unifor`, define a pasta correta `trashnet_mlp_cnn_unifor` e confere se o ZIP do dataset existe. Não é necessário montar o Google Drive nem alterar caminhos.
4. Execute a instalação das dependências e a verificação do TensorFlow/GPU. Só prossiga com GPU se o TensorFlow listar um dispositivo e o teste indicar `/device:GPU:0`. Se não houver GPU, altere o tipo de ambiente e reinicie o runtime.
5. Execute a preparação dos dados. O programa verifica o SHA-256, as 2.527 imagens, as seis classes, dimensões, arquivos corrompidos e duplicatas; depois cria a divisão estratificada 70%/15%/15% com seed 42 e as Figuras 1–4. Confira o relatório antes de treinar.
6. Execute a célula E3. É um treinamento completo para testar o fluxo, não uma execução de uma época.
7. Execute a célula de todos os experimentos. Ela usa `--skip-existing`, então E3 não será repetido se o modelo e a configuração tiverem sido salvos corretamente; E1–E6 serão concluídos.
8. Confira se E1, E2, E3, E4, E5 e E6 aparecem como OK na verificação. Só então execute `!python -m src.evaluate`.
9. Confira as três tabelas CSV e as Figuras 5–8. Por fim, execute a célula de backup e baixe `TrashNet_outputs_backup.zip` para o computador.

### Atenção à sessão do Colab

- O notebook não usa Google Drive. O repositório, dataset extraído, modelos, checkpoints, históricos, métricas e figuras ficam na VM temporária em `/content`.
- Se a sessão for encerrada ou o runtime for recriado, os arquivos de `outputs/` podem desaparecer. Baixe o backup antes de desconectar.
- Se precisar continuar depois de uma desconexão, envie o backup para o runtime novo e restaure a pasta `outputs` dentro de `/content/trashnet-mlp-cnn-unifor/trashnet_mlp_cnn_unifor/` antes de executar os treinamentos. Se não tiver backup, será necessário treinar novamente.
- O clone usa a branch `main`; quando o repositório for atualizado, abra um runtime novo ou execute novamente a primeira célula para obter a versão atual.

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
