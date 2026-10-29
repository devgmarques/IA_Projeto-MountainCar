# IA_Projeto_MountainCar
Projeto de Inteligência Artificial da UFU: Aprendizagem por reforço no ambiente MountainCar do Gymnasium

Aluno: Gustavo Marques Oliveira

# Problema
Controlar um carro com motor limitado para alcançar a bandeira na encosta direita de um vale. O agente precisa aprender a coordenar os impulsos e ganhar embalo, reduzindo o número de passos até a meta.

Usaremos o ambiente existente MountainCar-v0, da biblioteca Gymnasium, na configuração padrão com ações discretas. O projeto desenvolverá o agente, o treinamento e a avaliação, aproveitando a simulação pronta.
Ambiente e estado markoviano
O vale e os parâmetros físicos são fixos. O carro começa com velocidade zero e posição sorteada próxima ao fundo do vale.

O estado observado é s = (posição, velocidade).
- Posição: localização horizontal do carro.
- Velocidade: rapidez e sentido do movimento.
Com a física fixa, essas informações e a ação determinam o próximo estado, sem recuperar toda a trajetória. A posição sozinha não basta: dois carros no mesmo lugar podem estar se movendo em sentidos diferentes.

O agente recebe dois números contínuos; a imagem do ambiente serve para visualização.
# Ações
Código	Ação
0	Aplicar força do motor para a esquerda
1	Não aplicar força do motor
2	Aplicar força do motor para a direita

Uma ação por passo. Força do motor zero não faz o carro parar: a velocidade atual e a gravidade continuam atuando.
# Recompensa original

-1 por passo, inclusive no passo que alcança a meta. Não há bônus positivo de chegada.

Sem desconto, o retorno de uma tentativa de T passos é G = -T. Por exemplo, atingir a meta em 120 passos rende -120, melhor que -180 para 180 passos. São exemplos hipotéticos, não resultados de treinamento.

# Fim da tentativa
- Sucesso: posição x >= 0.5 e velocidade v >= 0, na configuração padrão (goal_velocity=0).
- Truncamento: limite externo de 200 passos.
O limite é uma proteção externa da execução. Caso o prazo faça parte do objetivo de um MDP de horizonte finito, o tempo restante deverá integrar o estado. Truncamento não é sinônimo de sucesso.

# Referências
Mountain Car documentação do Gymnasium: https://gymnasium.farama.org/environments/classic_control/mountain_car/

Código oficial do ambiente: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/mountain_car.py
