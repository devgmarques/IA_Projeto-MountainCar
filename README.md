# IA_Projeto-Taxi
Projeto de Inteligência Artificial da UFU: Aprendizagem por reforço no ambiente Taxi do Gymnasium

Problema
Treinar um agente para buscar um passageiro e desembarcá-lo no destino correto no ambiente Taxi do Gymnasium, reduzindo passos desnecessários e ações ilegais.
Ambiente e estado markoviano
Mapa fixo 5 × 5, com paredes e quatro pontos de atendimento. Configuração padrão, sem chuva nem mudança de destino.

Estado: (linha do táxi, coluna do táxi, localização do passageiro, destino).
O passageiro pode estar em um dos quatro pontos ou dentro do táxi. Com esse estado e as regras fixas, a próxima transição e a recompensa dependem do estado atual e da ação, sem consultar o trajeto anterior.
Ações
Mover para sul, norte, leste ou oeste; embarcar; desembarcar. Embarque e desembarque são ações explícitas.
Recompensas originais
- +20 por entrega correta.
- -10 por tentativa ilegal de embarcar ou desembarcar.
- -1 nos demais passos, inclusive movimento bloqueado por parede.
As categorias são alternativas, não cumulativas. Um desembarque em outro ponto designado pode ser permitido sem concluir a tarefa; não é automaticamente uma ação ilegal.
Término
A tarefa termina com a entrega correta. O limite externo padrão de 200 passos trunca tentativas longas; esse evento não conta como sucesso.

Referência
Taxi - documentação oficial do Gymnasium
