# Resultados finais do TrashNet

## Experimentos obrigatórios E1–E6

Os seis experimentos foram executados em CPU com TensorFlow/Keras. Após a conferência de três cópias binárias exatas com rótulos divergentes, o conjunto utilizado ficou com 2.524 imagens: glass 501, paper 594, cardboard 403, plastic 480, metal 409 e trash 137. A divisão estratificada (seed 42) contém 1.766 imagens de treino, 379 de validação e 379 de teste.

| Exp. | Modelo | Entrada | Acurácia teste | F1 macro | Épocas | Tempo total |
|---|---|---:|---:|---:|---:|---:|
| E1 | MLP | 128×96 | 23,48% | 6,35% | 11 | 151,55 s |
| E2 | MLP com pesos | 128×96 | 6,33% | 2,87% | 11 | 155,85 s |
| E3 | CNN | 128×96 | 76,78% | 74,76% | 49 | 1.501,60 s |
| E4 | CNN com pesos | 128×96 | 66,49% | 65,23% | 60 | 1.852,85 s |
| E5 | CNN | 256×192 | 68,87% | 65,58% | 60 | 8.048,93 s |
| E6 | CNN | 512×384 | 67,02% | 62,17% | 60 | 37.862,69 s |

E3 foi selecionado pela macro-F1 de validação (76,31%). Os arquivos detalhados estão em `outputs/resultados_finais/`.

## Extensão YOLO

YOLO26n-cls pré-treinado, entrada 224, lote 32, até 60 épocas e paciência 10. O histórico recuperado registra 28 épocas e execução em CPU. No teste: 90,50% de acurácia, 90,37% de precisão macro, 87,63% de recall macro e 88,72% de F1 macro. O histórico original não contém acurácia de treino por época; não foi reconstruída.

A comparação com E3 é indicativa, não isolada por arquitetura: YOLO usou pesos pré-treinados e resolução 224×224; E3 foi inicializada aleatoriamente e usou 128×96. As mesmas partições e imagens de teste foram mantidas.

## Experimento complementar AMD DirectML

A CNN E3 foi portada para PyTorch e treinada na AMD Radeon RX 5500 XT com DirectML. Foram concluídas 60 épocas; a melhor época de validação foi a 54. No teste, obteve 74,41% de acurácia e 71,78% de F1 macro. O tempo registrado foi 490,27 s. O monitoramento reuniu 280 amostras, com utilização média registrada de 39,05% e pico de 50%. É uma experiência complementar e não substitui E3 oficial, pois a implementação e o redimensionamento diferem.
