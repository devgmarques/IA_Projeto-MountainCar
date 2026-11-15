# 01 · Comece por aqui

**MountainCar com DQN — Desafio 2 de Inteligência Artificial**

Gustavo Marques Oliveira · UFU · Material preparado em 11/11/2026.

O objetivo é treinar uma rede neural para controlar o carro, mostrar o agente treinado funcionando e apresentar os resultados em dois slides. Você pode realizar tudo em CPU: o projeto não exige placa de vídeo dedicada.

O pacote contém código completo em Python/PyTorch, um treinamento executado, pesos salvos, dados de cada tentativa, gráficos, dois slides e um vídeo de reserva. Os resultados foram produzidos no ambiente de preparação deste material, com assistência de IA. **Ainda não foram executados no seu computador.** O roteiro abaixo permite repetir o experimento e entender cada parte.

| Teste separado: 100 tentativas por agente | Aleatório | DQN |
| --- | --- | --- |
| Chegadas à meta | 0/100 | 100/100 |
| Retorno médio | −200,00 | −100,44 |
| Passos médios nas tentativas bem-sucedidas | Não houve sucesso | 100,44 |

São resultados reais de **um treinamento**, não uma garantia de sucesso em todo treinamento futuro. O aleatório esgotou os 200 passos em todas as tentativas avaliadas.

## Sua ordem de trabalho

1. Extraia o ZIP e prepare o VS Code: páginas 2–3.
2. Rode primeiro o modelo já treinado: página 4. Isso confirma que a demonstração funciona no seu computador.
3. Faça seu próprio treinamento, avaliação e gráficos: página 5.
4. Estude a explicação do ambiente e do DQN: páginas 7–10.
5. Ensaie a apresentação de dois slides: páginas 11–12.
6. Atualize o GitHub com os arquivos corretos: páginas 13–14.
7. Confira a lista final e as referências: páginas 15–16.

**Para a aula:** deixe o modelo treinado salvo, os slides abertos e o comando de demonstração pronto. O treinamento deve acontecer antes da apresentação. Depois da instalação, a demonstração local funciona sem internet.

O repositório remoto não foi alterado automaticamente. As instruções deste guia mostram exatamente como atualizar o seu repositório existente.

# 02 · Preparar o Windows e o VS Code

## Instale o Python compatível

Este pacote usa **Python 3.12 de 64 bits**. As versões fixadas das bibliotecas foram escolhidas para esse ambiente. Se você já tem outro Python, pode manter os dois instalados.

1. Abra https://www.python.org/downloads/release/python-31210/ .
2. Na seção Files, escolha **Windows installer (64-bit)**. A versão 3.12.10 é indicada aqui por disponibilizar esse instalador; não é a versão mais recente do Python.
3. Execute o instalador. Mantenha o Python Launcher e marque a opção de adicionar o Python ao PATH, quando aparecer.
4. Feche e reabra o VS Code para ele reconhecer a instalação.

## Abra a pasta certa

1. Baixe `MountainCar_Desafio2_Completo.zip` e escolha **Extrair tudo** no Windows. Não execute arquivos dentro do ZIP.
2. Dentro da pasta extraída, entre em `IA_Projeto-MountainCar` e depois em `desafio2`.
3. No VS Code, escolha **File > Open Folder / Arquivo > Abrir Pasta** e abra essa pasta `desafio2`.
4. Instale a extensão **Python**, publicada pela Microsoft, se ainda não estiver instalada.
5. Use **Terminal > New Terminal / Novo Terminal**. Os comandos deste guia são para o PowerShell do Windows.

No terminal, execute:

```powershell
py -3.12 --version
Get-Location
dir
```

Você deve ver Python 3.12.x e arquivos como `dqn.py`, `treinar.py` e `requirements.txt`. Se aparecer apenas a pasta `desafio2`, você abriu a pasta de cima: execute `cd desafio2`.

**Pasta de trabalho:** os comandos das páginas seguintes são executados dentro de `desafio2`. Os caminhos relativos, como `runs/referencia`, começam nessa pasta.

Se seu Windows for ARM ou seu computador usar outro sistema operacional, os instaladores e comandos precisam ser adaptados. O roteiro principal é para Windows x64.

# 03 · Instalar as bibliotecas

O ambiente virtual `.venv` é uma pasta que guarda as bibliotecas deste projeto. Ele evita misturar dependências de trabalhos diferentes. Internet é necessária durante a instalação.

No terminal do VS Code, dentro de `desafio2`, rode **um comando de cada vez** e espere terminar:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe verificar.py
```

O último comando deve terminar com **OK: rede, ambiente e recompensa original funcionando**. A primeira instalação pode demorar para baixar o PyTorch.

## Escolha o interpretador do VS Code

Pressione `Ctrl+Shift+P`, procure **Python: Select Interpreter** e selecione `desafio2/.venv/Scripts/python.exe`. Se ele não aparecer, use **Enter interpreter path** e localize o arquivo.

Os comandos deste guia já usam o caminho completo desse Python. Portanto, não é necessário ativar o ambiente virtual nem alterar a política de execução do PowerShell. Use o terminal, em vez do botão Run do Code Runner.

**Alternativa automática:** depois de instalar o Python 3.12, você pode executar `INSTALAR_WINDOWS.bat`, que faz a mesma instalação e verificação. Escolha o procedimento manual ou o arquivo automático; não precisa realizar ambos.

Gymnasium fornece a simulação; PyTorch cria e treina a rede; NumPy manipula números; Matplotlib cria gráficos; Pygame mostra a janela; ImageIO grava o vídeo. O agente DQN está implementado nos arquivos deste projeto.

# 04 · Mostrar o agente que já está treinado

## Demonstração com janela

Dentro de `desafio2`, execute:

```powershell
.\.venv\Scripts\python.exe demonstrar.py
```

O programa carrega `runs/referencia/melhor_modelo.pt` e abre a janela do MountainCar. Serão três tentativas. Os pesos ficam fixos: **mostrar o agente não é treinar novamente**.

Na execução de preparação, essas tentativas chegaram à meta em 107, 106 e 106 passos, usando as seeds 30000, 30001 e 30002. O terminal imprime `SUCESSO` ou `LIMITE DE PASSOS`, o número de passos e o retorno. Confira o comportamento no seu computador antes da aula.

Para ver a referência aleatória em uma tentativa:

```powershell
.\.venv\Scripts\python.exe demonstrar.py --modo aleatorio --episodios 1
```

O agente aleatório sorteia esquerda, força zero ou direita. Ele não consulta os valores da rede. Para interromper, volte ao terminal e pressione `Ctrl+C`.

## Vídeo e execução sem janela

O arquivo `runs/referencia/demonstracao.mp4` já contém uma gravação real do modelo fornecido. Leve esse vídeo como reserva. A apresentação principal deve mostrar o programa ao vivo, conforme o pedido do professor.

Para conferir a execução apenas pelo terminal:

```powershell
.\.venv\Scripts\python.exe demonstrar.py --sem-janela
```

Para gravar novamente, sem abrir janela:

```powershell
.\.venv\Scripts\python.exe demonstrar.py --video runs/referencia/novo_video.mp4
```

**O arquivo `.pt`** contém os pesos aprendidos, a configuração e os dados do checkpoint. A função `load_model` recria a arquitetura e carrega esses pesos. Não precisa baixar modelos da internet durante a demonstração.

# 05 · Fazer seu próprio treinamento

Use uma pasta nova para preservar os resultados fornecidos. Esta é a sequência completa, sempre dentro de `desafio2`:

```powershell
.\.venv\Scripts\python.exe treinar.py --saida runs/meu_treino --passos 250000 --seed 42
```

Espere aparecer **Concluído**. A cada 50 episódios há um registro no terminal; a cada 10 mil passos ocorre uma validação. O treino não abre a janela do carro, para executar mais rápido.

Depois, avalie o modelo escolhido em 100 tentativas novas:

```powershell
.\.venv\Scripts\python.exe avaliar.py --pasta runs/meu_treino --episodios 100
```

Crie os gráficos e os dois slides PDF com seus próprios números:

```powershell
.\.venv\Scripts\python.exe graficos.py --pasta runs/meu_treino
```

Mostre o modelo que você acabou de treinar:

```powershell
.\.venv\Scripts\python.exe demonstrar.py --modelo runs/meu_treino/melhor_modelo.pt
```

Abra `runs/meu_treino/slides_resultados.pdf`. Esse arquivo é atualizado automaticamente a partir dos seus dados. Leia também `resultados_para_fala.txt`. **O PPTX fornecido continua mostrando o treinamento de referência**: se apresentar seu treino local, use o novo PDF ou atualize os números e gráficos do PPTX.

O treinamento de referência levou aproximadamente 71 segundos nesta máquina Linux/CPU. Isso não prevê o tempo no seu computador. A instalação é uma etapa separada. Mesmo com a mesma seed, plataformas e versões diferentes podem produzir resultados diferentes.

Para repetir, use outro nome, como `runs/meu_treino_2`. O programa recusa uma pasta já preenchida para evitar sobrescrever resultados. `Ctrl+C` salva pesos disponíveis, mas esta versão não retoma exatamente um treinamento interrompido: ela não salva o estado do otimizador e do replay para retomada.

Um teste curto com `--passos 2000 --saida runs/teste_instalacao` serve apenas para verificar a execução; não espere um agente competente com tão pouca experiência.

# 06 · Entender os arquivos do projeto

Na raiz, as apresentações do Desafio 1 foram preservadas. O trabalho novo fica em `desafio2`.

| Arquivo ou pasta | Para que serve |
| --- | --- |
| `dqn.py` | Rede neural, memória de experiências, atualização e pesos. |
| `treinar.py` | Interage com o ambiente, aprende e escolhe o melhor modelo pela validação. |
| `avaliar.py` | Compara DQN e aleatório sem aprender durante o teste. |
| `demonstrar.py` | Abre a janela do carro ou grava vídeo do modelo salvo. |
| `graficos.py` | Gera PNGs, dois slides PDF e um resumo para a fala. |
| `verificar.py` | Confere bibliotecas, rede e uma transição do ambiente. |
| `requirements.txt` | Fixa versões das dependências; PyTorch é instalado separadamente. |
| `tests/test_dqn.py` | Testes do alvo de Bellman, replay e carregamento. |
| `apresentacao/` | Dois slides prontos em PDF e PowerPoint. |
| `runs/referencia/` | Experimento executado durante a preparação do material. |
| `runs/meu_treino/` | Seus resultados, criados quando você treinar localmente. |

Dentro de uma pasta de experimento:

- `melhor_modelo.pt`: pesos selecionados pela maior média de retorno na validação. Use este na apresentação.
- `ultimo_modelo.pt`: pesos ao final da execução; podem ser piores que os anteriores.
- `treino.csv`: uma linha por episódio completo de treino, com retorno, passos, sucesso, epsilon e erro médio.
- `validacao.csv`: uma linha por avaliação periódica, com retorno e taxa de sucesso.
- `avaliacao.csv` e `resumo.json`: tentativas individuais e medidas do teste final.
- `config.json` e `execucao.json`: parâmetros, versões, duração e identificação da execução.
- `graficos/` e `slides_resultados.pdf`: figuras e apresentação geradas com os dados dessa pasta.

Os CSVs podem ser abertos no Excel. Se todas as colunas aparecerem juntas, use **Dados > De Texto/CSV** e escolha vírgula como separador. Ponto nos dados representa separador decimal.

# 07 · Entender o problema antes do algoritmo

**Gymnasium** é a biblioteca que oferece o ambiente pronto: física do carro, regras, ações, observações e recompensas. Nosso código escolhe as ações e aprende; não precisamos programar a física do jogo.

**Agente** é o programa controlador. **Estado** é a informação disponível para decidir. **Passo** é uma ação seguida de uma resposta do ambiente. **Episódio** é uma tentativa inteira, do reinício até chegar à meta ou alcançar o limite.

O estado tem dois números: posição horizontal e velocidade. Velocidade positiva indica movimento para a direita; negativa, para a esquerda; zero significa repouso naquele instante. A ação do motor não determina imediatamente o sentido do movimento: inércia e gravidade também atuam.

| Ação | Significado |
| --- | --- |
| 0 | Aplicar força do motor para a esquerda. |
| 1 | Não aplicar força do motor; o carro continua sujeito à física. |
| 2 | Aplicar força do motor para a direita. |

O motor não tem força suficiente para subir diretamente. O agente precisa aproveitar o movimento no vale para ganhar embalo. A regra de balançar não foi escrita como uma política manual no nosso agente.

## Recompensa e fim da tentativa

Cada passo vale **−1**, inclusive o passo da chegada. Não adicionamos bônus, penalidades por posição ou outra recompensa auxiliar. O sucesso acontece quando a posição é pelo menos 0,5 e a velocidade atende ao padrão do ambiente, que é pelo menos zero.

Sem desconto, o retorno é a soma das recompensas: 100 passos dão −100; 180 passos dão −180. Como −100 é maior, chegar mais rápido é melhor. Uma tentativa interrompida aos 200 passos soma −200. Chegar exatamente no passo 200 também pode somar −200: por isso registramos o indicador de sucesso separadamente.

O limite de 200 passos reinicia a tentativa, mas é um corte externo da simulação (`truncated`). A chegada é uma terminação da tarefa (`terminated`). Essa diferença entra no cálculo do DQN, explicado na página 9.

Com a física fixa, posição e velocidade bastam para a evolução física do próximo estado. Se o objetivo fosse definido com um prazo finito como parte da tarefa, o tempo restante também precisaria integrar o estado. Aqui tratamos os 200 passos como limite externo.

# 08 · A rede neural escolhida

Usamos uma **MLP**, rede de camadas totalmente conectadas. Ela é adequada porque a observação já contém dois números; não precisamos extrair informações das imagens.

| Parte da rede | Quantidade | Função |
| --- | --- | --- |
| Entrada | 2 valores | Posição e velocidade. |
| Camada oculta 1 | 64 neurônios + ReLU | Aprende combinações das entradas. |
| Camada oculta 2 | 64 neurônios + ReLU | Aprende relações mais complexas. |
| Saída linear | 3 valores Q | Uma estimativa para cada ação. |

São **4.547 parâmetros treináveis**, entre pesos e vieses. Eles começam aleatórios e são ajustados com as experiências. ReLU transforma entradas negativas em zero e mantém as positivas, permitindo relações não lineares. A saída não usa softmax: os valores Q não são probabilidades.

Antes de entrar na rede, os dois números são reescalados para uma faixa de −1 a 1 usando os limites conhecidos do ambiente. Isso evita que as diferenças de escala entre posição e velocidade dominem a entrada. Não usamos estatísticas do teste nessa normalização.

## O que significa Q?

`Q(estado, ação)` estima o retorno futuro com desconto ao escolher essa ação e continuar seguindo uma estratégia de alto valor. A rede fornece três estimativas em uma única chamada.

**Exemplo apenas ilustrativo:** se as saídas forem `[−70, −65, −60]`, a ação escolhida é 2, porque −60 é o maior valor. Isso não significa 60% de chance de acerto nem exatamente 60 passos restantes: usamos desconto de 0,99 e a previsão da rede é aproximada.

O treinamento ajusta os pesos para aproximar previsões e alvos calculados a partir da experiência. Na demonstração, usamos somente a previsão: estado entra, três valores saem, a ação de maior valor é executada.

No código, leia `QNetwork` em `dqn.py`: `forward` normaliza e calcula os valores; `action` escolhe o índice do maior valor com `argmax`.

# 09 · Como o DQN aprende com recompensa −1

**DQN = Deep Q-Network.** É Q-learning usando uma rede neural para aproximar a função Q. O sinal útil está também no que acontece depois da ação, não apenas na recompensa imediata.

1. O agente observa posição e velocidade.
2. Escolhe uma ação; no treino, às vezes explora sorteando uma ação.
3. Recebe −1, o próximo estado e os indicadores de fim.
4. Guarda essa experiência no replay buffer.
5. Sorteia um lote de experiências anteriores.
6. Compara a previsão da rede com um alvo de Bellman e ajusta os pesos.

## O alvo de Bellman usado no projeto

```text
y = recompensa + gamma × (1 − terminated) × max Q_alvo(próximo_estado, ação)
```

Se não chegou à meta e a maior previsão futura for −20, o alvo será `−1 + 0,99 × (−20) = −20,8`. Se chegou, `terminated = 1` e o alvo será apenas −1, pois não há mais passos da tarefa após a chegada. São exemplos numéricos ilustrativos.

À medida que experiências de sucesso são aprendidas, estados anteriores que levam a elas podem receber previsões melhores. Assim, a rede pode distinguir caminhos que geram menos custos futuros, apesar de cada passo isolado valer −1. Descobrir esses sucessos exige exploração, e o aprendizado pode oscilar.

**Truncamento:** ao atingir 200 passos sem chegar, reiniciamos o episódio, mas mantemos a parcela de valor futuro no alvo. Armazenamos o último estado observado antes de reiniciar. O limite externo não é tratado como chegada. Isso evita ensinar que esgotar o tempo encerra o custo da tarefa da mesma forma que alcançar a meta.

**Replay buffer:** guarda até 50 mil transições e sorteia lotes de 64. Reaproveita experiências e reduz a dependência entre amostras consecutivas.

**Rede alvo:** uma cópia da rede principal, atualizada a cada mil passos. Ela fornece a previsão futura com parâmetros temporariamente fixos, reduzindo mudanças bruscas no alvo.

**Huber e Adam:** Huber mede a diferença entre previsão e alvo; Adam ajusta os pesos para reduzir essa diferença. Limitamos a norma do gradiente a 10. O código é DQN básico, sem Double DQN ou replay priorizado.

# 10 · Como o experimento foi organizado

| Parâmetro | Valor usado |
| --- | --- |
| Orçamento e seed do treino | 250.000 passos; seed 42 |
| Rede / otimizador / taxa de aprendizado | 2–64–64–3; Adam; 0,001 |
| Gamma / perda | 0,99; Huber |
| Replay / lote | 50.000 transições; 64 amostras |
| Início e frequência das atualizações | Após 2.000 passos; a cada 4 passos |
| Cópia para a rede alvo | A cada 1.000 passos |
| Exploração epsilon-greedy | De 1,0 a 0,05 nos primeiros 75.000 passos |
| Validação | A cada 10.000 passos; 10 episódios fixos |

**Epsilon** é a probabilidade de sortear uma ação no treino. Começa em 100% e cai até 5%. Nas demais vezes, escolhemos o maior Q. Na validação, no teste e na demonstração, não há exploração aleatória do DQN.

**Treino:** coleta experiência e modifica os pesos. Foram concluídos 1.657 episódios; sobraram 77 passos de um episódio parcial ao atingir o orçamento. A seed controla a inicialização aleatória, sem garantir igualdade numérica entre plataformas.

**Validação:** testa periodicamente 10 condições iniciais fixas, seeds 10000–10009, sem modificar pesos. O melhor modelo foi escolhido no passo 210.000, com retorno médio −100,3. O último modelo não precisa ser o melhor.

**Teste:** só depois da seleção, comparamos os agentes em 100 condições iniciais novas, seeds 20000–20099. Ambos recebem as mesmas posições iniciais e o limite de 200 passos. O aleatório sorteia as três ações uniformemente; o DQN usa os pesos fixos. O teste não participou da seleção do modelo.

O gráfico de validação mostra o avanço do treinamento sem o ruído extra das ações exploratórias. A curva sobe quando o retorno fica menos negativo; as quedas mostram instabilidade. O gráfico bruto de treino e os diagnósticos também estão no pacote.

**Limite da conclusão:** 100 testes de um modelo não são 100 treinamentos independentes. O resultado mostra melhora sobre o aleatório neste experimento. Para avaliar estabilidade entre treinamentos, uma próxima etapa seria repetir com várias seeds de treino e apresentar a variação.

# 11 · Roteiro para apresentar os dois slides

Duração sugerida: cerca de 3–4 minutos de fala, mais a demonstração. Leia para entender e depois explique com suas palavras. Os números abaixo pertencem ao experimento de referência; substitua pelos seus se repetir o treino.

## Slide 1 — Problema, rede e curva de aprendizado

“Meu problema é o MountainCar. O carro tem um motor fraco e precisa ganhar embalo para alcançar a bandeira. O agente recebe posição e velocidade e escolhe entre força para a esquerda, força zero e força para a direita. Mantive a recompensa original de menos um por passo.”

“Implementei um DQN em PyTorch com duas camadas ocultas de 64 neurônios. A saída tem três valores Q, um para cada ação. Durante o treinamento, o agente explora, armazena experiências e ajusta a rede usando lotes dessas experiências e uma rede alvo.”

“O treinamento usou 250 mil passos. Neste gráfico, cada ponto é o retorno médio em dez episódios de validação sem exploração. Quanto menos negativo, melhor. A evolução oscilou. Escolhi o modelo com melhor validação, no passo 210 mil, antes de fazer o teste final.”

Aponte os eixos: horizontal = quantidade acumulada de passos de treino; vertical = soma média das recompensas por episódio na validação.

## Slide 2 — Resultados e comparação

“Comparei o modelo selecionado com um agente que apenas sorteia ações. Foram cem tentativas por agente, usando as mesmas condições iniciais e sem atualizar os pesos. O aleatório não chegou à meta em nenhuma dessas tentativas. O DQN chegou nas cem, com média de 100,44 passos.”

“O retorno médio foi menos 100,44 para o DQN e menos 200 para o aleatório. Como cada passo custa menos um, o resultado do DQN indica que chegou antes do limite. Esses números descrevem um único treinamento; não garantem que qualquer nova execução terá o mesmo desempenho.”

## Demonstração ao vivo

“Agora vou carregar os pesos salvos e mostrar o agente decidindo pelas previsões da rede. Nesta etapa ele não está aprendendo.”

Alterne para o terminal do VS Code e execute `demonstrar.py` com o comando da página 4, ou aponte `--modelo` para o seu treinamento. Mostre a janela e o registro de sucesso no terminal. Se houver problema na janela, explique e use o vídeo de reserva como evidência complementar.

Se questionado sobre a preparação, seja transparente: o material e a implementação foram preparados com assistência de IA; indique quais execuções você realizou e consegue explicar. Não atribua à sua máquina uma execução que ocorreu no ambiente de preparação.

# 12 · Perguntas que você deve saber responder

**Por que você escolheu essa rede?** A observação tem só dois números. Uma MLP pequena consegue aproximar relações não lineares entre posição, velocidade e valor das ações. Não foi necessário usar uma rede para imagens.

**Você programou o carro para ir para trás e depois para frente?** Não. O código fornece as ações possíveis e o processo de aprendizado. Os pesos aprendidos determinam a decisão a partir do estado; não há uma regra manual de movimento usada como agente.

**Como sabe que uma ação é boa se a recompensa sempre vale −1?** O DQN estima também os custos futuros. Chegar encerra a tarefa, enquanto continuar andando acumula custos. Experiências de chegada ajudam a melhorar as previsões de ações que levam até ela.

**Por que uma rede alvo e um replay?** A rede alvo mantém a previsão usada no alvo fixa por um intervalo; o replay reaproveita experiências em lotes misturados. Ambos ajudam a estabilizar as atualizações.

**O que são os três números na saída?** Estimativas de retorno futuro com desconto, uma por ação. Escolhemos o maior. Não são probabilidades e não precisam somar um.

**O que significa gamma 0,99?** Uma recompensa um passo à frente pesa 0,99; duas à frente, 0,99². O treinamento usa esse desconto. Já o retorno dos gráficos é a soma comum das recompensas do episódio, sem desconto.

**Por que o gráfico piora em alguns momentos?** Alterar os pesos pode melhorar algumas decisões e piorar outras. O aprendizado não é monotônico. Por isso acompanhamos validação e salvamos o melhor checkpoint, em vez de assumir que o último é superior.

**O aleatório levou 200 passos para chegar?** Não. Ele não chegou em nenhum dos cem testes: 200 é o limite que interrompeu as tentativas. A média de passos até a meta dele é indefinida nesse conjunto.

**Cem por cento de sucesso significa que resolveu para sempre?** Não. Significa cem sucessos nas cem condições avaliadas deste modelo. Ainda faltam repetições independentes do treinamento e testes mais amplos para avaliar robustez.

**Qual é a diferença entre treinar e demonstrar?** Treinar modifica pesos com experiências. Demonstrar carrega pesos já aprendidos e apenas calcula ações. É o modelo selecionado, e não necessariamente o último modelo, que deve ser carregado.

**O DQN é supervisionado?** O problema é aprendizado por reforço: as experiências vêm das ações do agente. Dentro da atualização há uma regressão para aproximar um alvo calculado pela equação de Bellman, sem uma lista de ações corretas fornecida por uma pessoa.

# 13 · Atualizar o GitHub pelo navegador

Seu repositório continua sendo `devgmarques/IA_Projeto-MountainCar`. **Não precisa excluir o Desafio 1 nem criar outro repositório.** O pacote já contém o README atualizado e os arquivos antigos preservados.

## Caminho mais simples

1. Extraia uma cópia limpa do ZIP, separada da pasta onde instalou as bibliotecas. Isso evita enviar `.venv` por engano.
2. Abra https://github.com/devgmarques/IA_Projeto-MountainCar . Entre na sua conta.
3. Na página principal dos arquivos, escolha **Add file > Upload files**.
4. De dentro de `IA_Projeto-MountainCar` no pacote limpo, arraste **README.md**, **.gitignore** e a pasta inteira **desafio2** para a área de envio. Não arraste a pasta externa `IA_Projeto-MountainCar`, pois ela criaria uma subpasta desnecessária no repositório.
5. Confira os caminhos: deve aparecer `desafio2/dqn.py`, por exemplo. O `README.md` da raiz substituirá a versão anterior. Os dois arquivos do Desafio 1 já existentes não precisam ser enviados outra vez.
6. Escreva a mensagem de commit: `Adiciona Desafio 2: DQN, modelo treinado e resultados`.
7. Escolha a opção de confirmar as alterações na branch principal e clique em **Commit changes**.
8. Volte à raiz e abra `desafio2/apresentacao/Desafio2_DQN.pdf`. Confira também se `runs/referencia/melhor_modelo.pt` e os CSVs aparecem.

O navegador do GitHub aceita até 100 arquivos por envio e 25 MiB por arquivo. Este pacote limpo fica abaixo desses limites. Não envie o ZIP como único arquivo: o professor precisa conseguir ver o código, os resultados e as apresentações organizados.

**Não envie:** `.venv`, `__pycache__`, instaladores do Python, dependências baixadas ou arquivos temporários. O `.gitignore` protege o fluxo com Git, mas não filtra automaticamente os arquivos que você arrasta no site. Por isso usamos uma cópia limpa.

## Se você treinou novamente no seu computador

Envie também sua pasta `desafio2/runs/meu_treino`, incluindo pesos, CSVs, JSONs, gráficos e `slides_resultados.pdf`. Edite o README para indicar qual experimento você está apresentando e atualizar a tabela de resultados. Prefira apresentar o PDF gerado nessa pasta: ele corresponde ao seu modelo.

Confirme que o professor consegue abrir o link. Se o repositório for privado, ele precisa de acesso; não basta enviar um endereço inacessível. A atualização remota é feita por você ao confirmar o commit no GitHub.

# 14 · Alternativa: atualizar com Git no VS Code

Esta página é opcional. Se o envio pelo navegador já funcionou, você pode pular. É necessário ter o Git instalado e autenticar sua conta quando o GitHub solicitar.

Em uma pasta de trabalho fora do pacote extraído:

```powershell
git clone https://github.com/devgmarques/IA_Projeto-MountainCar.git
cd IA_Projeto-MountainCar
```

Pelo Explorador de Arquivos, copie da distribuição limpa para a pasta clonada: `README.md`, `.gitignore` e a pasta `desafio2`. Substitua o README quando solicitado. Mantenha os arquivos do Desafio 1 e a pasta de controle do Git já existente.

De volta ao terminal, na raiz clonada:

```powershell
git status
git add README.md .gitignore desafio2
git diff --cached --stat
git commit -m "Adiciona Desafio 2 com DQN e resultados"
git push
```

Antes do commit, confira a lista mostrada: não deve incluir milhares de arquivos de dependências. Se for a primeira configuração do Git, ele pode pedir seu nome e e-mail; use os dados da sua conta. Não coloque senha ou token no código do projeto.

Se você já possui uma cópia Git local, use-a em vez de clonar novamente. Confira `git status` antes de copiar para não sobrescrever alterações que só existem nela. Se o push avisar que há alterações remotas, sincronize com `git pull --rebase` após salvar/commitar seu trabalho; **não use force push** para contornar o aviso. Conflitos precisam ser resolvidos antes de continuar.

## Como registrar o seu novo experimento

Depois de treinar, avaliar e gerar os gráficos, atualize a tabela e o link da apresentação no README. Então, da raiz:

```powershell
git add README.md desafio2/runs/meu_treino
git commit -m "Registra treinamento local e avaliacao do DQN"
git push
```

Não copie apenas um gráfico novo por cima de um resultado antigo. Modelo, avaliação e apresentação devem identificar o mesmo experimento. Mantenha a pasta de referência para comparação e use nomes separados para novas execuções.

# 15 · Resolver problemas e ensaiar

| Sintoma | O que conferir |
| --- | --- |
| `py` não foi reconhecido | Reabra o VS Code; confira a instalação do Python Launcher. |
| Não encontra Python 3.12 | Rode `py -0p`; instale 3.12 x64 e crie a `.venv` com `py -3.12`. |
| `No matching distribution found` | Confira Python 3.12 x64 e o comando de instalação. Não use Python 3.15 com estas versões fixadas. |
| `ModuleNotFoundError` | Use `.\.venv\Scripts\python.exe` e instale dependências nessa mesma `.venv`. |
| Não encontra `treinar.py` | Execute `dir`; o terminal precisa estar dentro de `desafio2`. |
| Modelo não encontrado | Confira se o ZIP foi extraído por inteiro e se `--modelo` aponta para um `.pt` existente. |
| Pasta de saída preenchida | Escolha outra pasta em `--saida`; o programa evita apagar o treino anterior. |
| Gráficos dizem que faltam arquivos | Complete o treino e rode `avaliar.py` antes de `graficos.py`. |
| Gráficos dizem que o modelo mudou | Avalie novamente o modelo atual antes de gerar os gráficos. |
| Janela não abre | Rode `verificar.py`, tente a demonstração com `--sem-janela` e confira a instalação do Pygame. |

Se uma `.venv` foi criada com o Python errado, feche os processos que a usam e renomeie apenas essa pasta para `.venv_antiga`. Crie outra com Python 3.12 e reinstale as dependências. Preserve `runs/`, onde estão seus resultados.

O caminho mais simples usa o Python da `.venv` diretamente, sem ativação. Uma mensagem sobre scripts bloqueados ao tentar `Activate.ps1` não exige mudar configurações de segurança para seguir este guia.

## Lista de conferência antes da aula

- Execute a demonstração no mesmo notebook que levará para a sala.
- Abra o PDF de dois slides e confirme que corresponde ao modelo escolhido.
- Deixe o terminal na pasta `desafio2`, com o comando de demonstração preparado.
- Tenha o vídeo MP4 salvo localmente e saiba onde abri-lo.
- Abra o link do GitHub e confira código, pesos e resultados.
- Saiba explicar estado, ações, recompensa, rede, replay, rede alvo e comparação.
- Ensaie a fala uma vez cronometrando. Confira o prazo e a ordem da apresentação na comunicação da turma.

Se quiser executar os testes do código: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. Eles verificam cálculos importantes; não substituem a avaliação de desempenho do agente.

# 16 · Evidências, fontes e limites

## O que foi verificado

O treinamento completo, a avaliação de 100 episódios por agente, o carregamento dos pesos, a demonstração sem janela e a gravação MP4 foram executados. Os três testes automatizados passaram. O caminho de renderização humana foi exercitado com driver de vídeo simulado; **a janela real no seu Windows ainda deve ser conferida por você**.

Ambiente de preparação: Linux x86_64, Python 3.12.14, PyTorch 2.5.1+cpu, Gymnasium 1.2.2 e NumPy 2.3.5. Os detalhes estão em `runs/referencia/execucao.json`. O pacote mantém os dados brutos para conferir os gráficos, em vez de apresentar números simulados.

O código é uma implementação didática de DQN, sem biblioteca pronta de agentes, sem política manual e sem alterar a recompensa original. A rede não aprende a partir do vídeo. As observações usadas são posição e velocidade.

Os cem testes avaliam um modelo selecionado pela validação. Não demonstram superioridade sobre outros algoritmos nem estabilidade entre várias inicializações do treinamento. Também não provam que o caminho aprendido é ótimo.

## Fontes oficiais para consultar

- Ambiente MountainCar, ações, observações, recompensa e limites: https://gymnasium.farama.org/environments/classic_control/mountain_car/
- Tutorial de DQN em PyTorch, para estudar os componentes: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Diferença entre terminação e truncamento: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
- Instalação de versões anteriores do PyTorch: https://pytorch.org/get-started/previous-versions/
- Ambiente virtual no Python: https://docs.python.org/3.12/library/venv.html
- Instalador Windows do Python 3.12.10: https://www.python.org/downloads/release/python-31210/
- Envio de arquivos ao GitHub: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

Os tutoriais são fontes conceituais e de API. Os resultados apresentados pertencem à execução deste projeto. A implementação, os materiais e os testes foram preparados com assistência de IA. Use o guia para executar, conferir e compreender o que será apresentado.

**Seu ponto de partida prático:** concluir a instalação, rodar `demonstrar.py` e depois seguir a sequência treino → avaliação → gráficos. Você não precisa decorar todo o código antes de conseguir ver o agente funcionando.
