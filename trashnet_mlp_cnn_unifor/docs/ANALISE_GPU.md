# Avaliação de GPU e custo computacional

## Por que usar GPU

As convoluções da CNN e as operações densas da MLP trabalham com matrizes/tensores. Em geral, uma GPU compatível pode acelerar essas operações paralelas em comparação com CPU. O ganho real depende do tipo de GPU, do lote, da resolução, da versão do TensorFlow, da leitura das imagens e do tempo de inicialização. Por isso, o projeto registra hardware e medições por experimento em vez de presumir um ganho fixo.

## Carga esperada por resolução

Cada imagem RGB em `float32` ocupa aproximadamente:

| Resolução | Valores por imagem | Memória de pixels por imagem |
|---|---:|---:|
| 128 × 96 × 3 | 36.864 | 0,14 MiB |
| 256 × 192 × 3 | 147.456 | 0,56 MiB |
| 512 × 384 × 3 | 589.824 | 2,25 MiB |

Esses números consideram apenas o tensor de entrada normalizado, não as ativações das camadas, gradientes, pesos, estados do otimizador, buffers ou cópias temporárias. A CNN de 512 × 384 pode, portanto, exigir muito mais que 2,25 MiB por imagem durante o treinamento.

Ao passar de 128 × 96 para 256 × 192, o total de pixels por imagem aumenta 4 vezes. Ao passar de 128 × 96 para 512 × 384, aumenta 16 vezes. O custo real da CNN não precisa crescer exatamente nessa proporção, porque depende das camadas, do lote e do hardware.

## Estratégias implementadas

1. **Leitura em lotes (`tf.data`):** evita manter todas as imagens de todas as resoluções na RAM ao mesmo tempo.
2. **`prefetch(AUTOTUNE)`:** permite preparar o próximo lote enquanto o modelo processa o atual, quando o runtime consegue sobrepor as operações.
3. **`set_memory_growth`:** evita reservar antecipadamente toda a VRAM disponível para o processo TensorFlow.
4. **Lotes adaptativos:** tenta 32, 16, 8, 4 e 2. Ao encontrar `ResourceExhaustedError`, registra a falha e tenta o próximo lote; a tentativa com o novo lote reinicia o experimento para preservar a comparação.
5. **`GlobalAveragePooling2D`:** reduz mapas de características a um vetor antes das camadas densas e evita que os parâmetros da camada final dependam diretamente de cada pixel espacial.
6. **Checkpoint por época:** permite retomar a execução após interrupção do runtime, sem salvar um backup a cada lote.

O tamanho do lote efetivamente usado é gravado na configuração de cada experimento. Caso a GPU não suporte o mesmo lote para E3, E5 e E6, a diferença deve ser descrita no artigo como ajuste por restrição de memória, conforme a proposta.

## O que o código mede

Os dados ficam em `outputs/metricas/config_E1.json` até `config_E6.json` e na `tabela5_custo.csv`.

| Medida | Origem | Como interpretar |
|---|---|---|
| GPU detectada e modelo | TensorFlow e `nvidia-smi` | Confirma qual dispositivo foi reconhecido. |
| Parâmetros totais e treináveis | Keras | Tamanho estrutural da rede, não é memória de VRAM. |
| Tempo total e médio por época | `time.perf_counter` e cronômetro por época | Inclui leitura/decodificação/redimensionamento dos lotes durante `fit`. |
| RAM máxima amostrada | RSS do processo via `psutil` | Não é o pico instantâneo absoluto entre amostras. |
| Pico do alocador GPU TensorFlow | `tf.config.experimental.get_memory_info` | Mede memória rastreada pelo alocador do TensorFlow, não necessariamente todo o uso de VRAM do dispositivo. |
| Pico de VRAM do processo | `nvidia-smi`, quando disponível | Amostras do processo Python; depende do driver expor PID e memória. |
| Utilização da GPU | `nvidia-smi`, quando disponível | Utilização do dispositivo como um todo, amostrada; pode incluir outros processos. |

As medições de RAM/VRAM são amostrais e podem perder picos muito curtos. Quando o método não está disponível, o campo é `null`/vazio. Não substituam valores ausentes por estimativas.

## Procedimento correto de avaliação

1. No Colab, selecionar runtime GPU.
2. Conferir `tf.config.list_physical_devices('GPU')` e executar o teste de multiplicação de matrizes na GPU.
3. Registrar modelo, versão TensorFlow, Python, lote e memória total da placa.
4. Rodar a mesma divisão e semente em todos os experimentos.
5. Comparar E3/E5/E6: acurácia e F1 macro, tempo médio por época, tempo total, lote, parâmetros e memória.
6. Se lotes diferirem, registrar a justificativa e evitar afirmar que a comparação foi feita com condições de lote idênticas.

## Colab e VS Code Windows

O Colab é a opção recomendada quando a máquina local não tem GPU compatível. A disponibilidade, o tipo de GPU e os limites de uso podem variar; selecionar GPU no menu não garante que a sessão recebeu uma placa específica. É obrigatório confirmar a detecção pelo TensorFlow.

No Windows nativo, TensorFlow 2.11 ou posterior não oferece suporte oficial à execução CUDA em GPU. Para GPU NVIDIA fora do Colab, a alternativa oficial é usar WSL2 com a configuração de driver/CUDA compatível. No Windows nativo, o projeto pode ser testado em CPU; o tempo será diferente e deve ser anotado.

## Texto-base honesto para o artigo (preencher após executar)

> O treinamento foi realizado em [modelo da GPU/CPU], com [versão] do TensorFlow e lote [valores por experimento]. O consumo de memória foi acompanhado pelo alocador do TensorFlow e, quando disponível, pelo `nvidia-smi`; por se tratar de coleta amostral, os valores representam os maiores registros observados durante a execução, não um monitoramento contínuo do hardware. O tempo de treinamento inclui o processamento dos lotes durante `model.fit`, mas não inclui download e inspeção inicial do conjunto de dados.

Substituir os campos entre colchetes por informações extraídas de `config_E*.json`. Não use este parágrafo sem preencher e conferir os dados.
