# Como interpretar a telemetria

O arquivo `gpu_monitoramento.csv` registra amostras durante a execução. As colunas principais são:

| Campo | O que representa |
|---|---|
| `gpu_engine_max_percent` | Maior percentual de utilização entre os motores GPU associados ao processo Python naquela amostra. |
| `gpu_process_dedicated_mb` | Memória dedicada atribuída ao processo Python. |
| `gpu_process_shared_mb` | Memória compartilhada atribuída ao processo Python. |
| `gpu_adapter_dedicated_mb` | Memória dedicada total do adaptador associado ao processo. |
| `gpu_adapter_shared_mb` | Memória compartilhada total desse adaptador. |
| `process_ram_mb` | RAM residente usada pelo processo Python. |
| `process_cpu_percent` | Uso de CPU do processo Python. |
| `elapsed_seconds` | Segundos desde o início do monitoramento. |

Os campos de GPU vêm dos contadores de desempenho do Windows. O monitor consulta o PID do próprio Python e os LUIDs de adaptador associados a esse processo. Isso ajuda a distinguir o uso do treinamento do uso de outros programas.

O valor de `gpu_engine_max_percent` não deve ser confundido com o percentual global mostrado no Gerenciador de Tarefas: ele representa o motor mais ocupado do processo na amostra. Um valor baixo isolado não prova que a GPU não foi usada; analise a série temporal, a memória do processo e o dispositivo DirectML registrado em `resumo_execucao.json`.

Se `counter_status` mostrar `OK`, a consulta ao Windows respondeu. Se aparecer `INDISPONÍVEL`, a amostra não tem dados confiáveis de GPU e o treinamento não deve começar até resolvermos a consulta aos contadores.
