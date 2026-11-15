# Proveniência do experimento de referência

Material preparado com assistência de IA em 11/11/2026 para Gustavo Marques Oliveira. Código, dados e resultados foram produzidos para este projeto. Não houve alteração remota do repositório durante a preparação do pacote.

## Base preservada

Repositório consultado: https://github.com/devgmarques/IA_Projeto-MountainCar . Commit de base: `d7440243730e09bde145945aadf8cf4c497e67d3`. Os arquivos `MountainCar_Desafio1.pdf` e `MountainCar_Desafio1.pptx` foram preservados. O README foi ampliado para incluir o Desafio 2.

## Execução real

- Comando: `python -u treinar.py --saida runs/referencia --passos 250000 --seed 42`.
- Ambiente: Linux x86_64, Python 3.12.14, PyTorch 2.5.1+cpu, Gymnasium 1.2.2 e NumPy 2.3.5.
- 250.000 passos, 1.657 episódios completos, 77 passos do episódio parcial final; cerca de 70,924 segundos neste equipamento. Esse tempo não é uma estimativa para outros computadores.
- Um único treinamento completo foi executado. Não foram escolhidas execuções a partir do desempenho no teste.
- Seleção do checkpoint: maior retorno médio entre avaliações em 10 seeds fixas de validação, 10000–10009. Melhor checkpoint: passo 210000, retorno médio −100,3.
- Avaliação separada: `python avaliar.py --pasta runs/referencia --episodios 100`, seeds 20000–20099. Sem aprendizagem, mesmas condições iniciais por agente. Ações aleatórias uniformes com gerador seed 12345.
- DQN: 100/100 sucessos; retorno médio −100,44; desvio padrão amostral 8,81610097; média de 100,44 passos até a meta.
- Aleatório: 0/100 sucessos; retorno médio −200. Não existe média de passos até a meta nesse conjunto, porque nenhuma tentativa teve sucesso.
- Vídeo: três episódios novos, seeds 30000–30002, com sucessos em 107, 106 e 106 passos. Gravação gerada pelo ambiente, sem animação artificial.

SHA-256 do modelo selecionado: `60a1e0c57589b9a3eda74cc0b25949339acc6676e9e69dac90464f70afedf83b`.

`resumo.json` registra o hash, passo do modelo e data da avaliação. `graficos.py` recusa gerar resultados se os pesos tiverem mudado depois da avaliação. `manifesto_sha256.json` registra os hashes do código principal e dos dados de referência distribuídos.

## Validação e limites

Passaram os testes do alvo de Bellman, consistência do replay circular e salvamento/carregamento da rede. A instalação, o treino, a avaliação, os gráficos e a gravação de vídeo foram executados em Linux/CPU. A renderização `human` foi exercitada com driver simulado; não se afirma que uma janela real foi testada no Windows do aluno.

O algoritmo é DQN básico, com rede alvo, replay uniforme, Huber e Adam. Recompensa original −1, sem reward shaping. O tratamento de truncamento mantém bootstrap, pois os 200 passos são um limite externo. O estado seguinte armazenado é o observado antes do reset.

Os intervalos de Wilson em `resumo.json` descrevem a incerteza da taxa de sucesso para este modelo sob amostragem das condições iniciais, e não variabilidade entre treinamentos. O experimento não prova ótimo global ou sucesso universal. Resultados locais devem ser registrados em outra pasta e apresentados com os próprios gráficos.
