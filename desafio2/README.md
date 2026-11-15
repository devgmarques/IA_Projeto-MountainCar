# Desafio 2 — DQN no MountainCar

Comece pelo **[GUIA_COMPLETO.md](GUIA_COMPLETO.md)** ou pelo **[PDF do guia](GUIA_COMPLETO.pdf)**. Eles incluem instalação no Windows, execução no VS Code, treinamento próprio, explicação do algoritmo, roteiro de fala e atualização do GitHub.

## Para apresentar

- [Dois slides PDF](apresentacao/Desafio2_DQN.pdf)
- [PowerPoint editável, com notas de apresentação](apresentacao/Desafio2_DQN.pptx)
- [Modelo treinado](runs/referencia/melhor_modelo.pt)
- [Vídeo de reserva](runs/referencia/demonstracao.mp4)

Os slides fornecidos usam os dados de `runs/referencia`, executados durante a preparação do material. Após seu próprio treino, use `graficos.py --pasta runs/meu_treino` para gerar um PDF de dois slides com seus resultados. Os arquivos de PowerPoint e PDF fornecidos têm o mesmo conteúdo experimental, com diferenças de disposição visual.

## Ordem dos programas

1. `verificar.py`: testar instalação.
2. `demonstrar.py`: conferir o modelo fornecido com uma janela.
3. `treinar.py --saida runs/meu_treino`: treinar novamente.
4. `avaliar.py --pasta runs/meu_treino`: comparar com ações aleatórias.
5. `graficos.py --pasta runs/meu_treino`: gerar gráficos e slides atualizados.
6. `demonstrar.py --modelo runs/meu_treino/melhor_modelo.pt`: mostrar seu modelo.

Preceda os comandos pelo Python da `.venv`, conforme o guia. Execute nesta pasta `desafio2`. Não precisa de GPU e não há treinamento durante a demonstração.
