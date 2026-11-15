"""Gera gráficos PNG, resumo textual e 2 slides PDF a partir dos CSVs reais."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BG, FG, MUTED, GOLD, CYAN = '#091322', '#F4F6FA', '#B9C6D7', '#F9C74F', '#43D9E6'


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED)
    for side in ['top', 'right']: ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']: ax.spines[side].set_color(MUTED)
    ax.grid(axis='y', alpha=.18, color=MUTED)
    ax.set_axisbelow(True)


def validation_plot(ax, val, selected):
    x = [float(r['step']) / 1000 for r in val]
    y = [float(r['mean_return']) for r in val]
    ax.plot(x, y, color=GOLD, marker='o', markersize=4, linewidth=2)
    ax.axvline(selected / 1000, color=CYAN, linestyle='--', alpha=.7, label='Modelo selecionado')
    ax.set(xlabel='Passos de treinamento (milhares)', ylabel='Retorno médio na validação', ylim=(-210, -70))
    style(ax)
    ax.legend(facecolor=BG, labelcolor=FG, edgecolor='none', fontsize=10)


def compare_plot(ax1, ax2, summary):
    names = ['Aleatório', 'DQN']
    values = [100 * summary[k]['success_rate'] for k in ['aleatorio','dqn']]
    ax1.bar(names, values, color=[MUTED, CYAN], width=.5)
    ax1.set(ylim=(0,115), ylabel='Tentativas com sucesso (%)')
    for i,v in enumerate(values): ax1.text(i, v+3, f'{v:.0f}%', ha='center', color=FG, fontsize=16, weight='bold')
    values = [summary[k]['mean_return'] for k in ['aleatorio','dqn']]
    ax2.bar(names, values, color=[MUTED, GOLD], width=.5)
    ax2.set(ylim=(-225,0), ylabel='Retorno médio (maior é melhor)')
    for i,v in enumerate(values): ax2.text(i, v-12, f'{v:.2f}'.replace('.',','), ha='center', color=FG, fontsize=14, weight='bold')
    style(ax1); style(ax2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pasta', default='runs/referencia')
    args = p.parse_args()
    folder = Path(args.pasta)
    required = ['treino.csv', 'validacao.csv', 'resumo.json', 'config.json', 'execucao.json', 'melhor_modelo.pt']
    missing = [f for f in required if not (folder/f).is_file()]
    if missing: p.error(f'Faltam {missing}. Execute treinar.py e avaliar.py antes.')
    summary = json.loads((folder/'resumo.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    execution = json.loads((folder/'execucao.json').read_text(encoding='utf-8'))
    if summary['protocol']['model_sha256'] != hashlib.sha256((folder/'melhor_modelo.pt').read_bytes()).hexdigest():
        p.error('O modelo mudou após a avaliação. Execute avaliar.py novamente.')
    train, val = rows(folder/'treino.csv'), rows(folder/'validacao.csv')
    selected = summary['protocol']['model_step']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'savefig.facecolor':BG, 'text.color':FG})
    graphs = folder/'graficos'; graphs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,5), facecolor=BG)
    step = np.array([float(r['step']) for r in train]) / 1000
    ret = np.array([float(r['return']) for r in train])
    ax.plot(step,ret,alpha=.22,color=CYAN,linewidth=.8,label='Cada episódio (com exploração)')
    if len(ret) >= 50:
        ax.plot(step[49:],np.convolve(ret,np.ones(50)/50,mode='valid'),color=GOLD,linewidth=2,label='Média móvel de 50 episódios')
    ax.set(xlabel='Passos de treinamento (milhares)',ylabel='Retorno por episódio',ylim=(-210,-65))
    style(ax);ax.legend(facecolor=BG,labelcolor=FG,edgecolor='none');fig.tight_layout()
    fig.savefig(graphs/'treinamento.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),facecolor=BG)
    validation_plot(ax,val,selected);fig.tight_layout();fig.savefig(graphs/'validacao.png',dpi=170);plt.close(fig)
    fig,(a,b)=plt.subplots(1,2,figsize=(11,5),facecolor=BG)
    compare_plot(a,b,summary);fig.tight_layout();fig.savefig(graphs/'comparacao.png',dpi=170);plt.close(fig)
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4),facecolor=BG)
    a.plot(step,[float(r['epsilon']) for r in train],color=GOLD)
    a.set(xlabel='Passos (milhares)',ylabel='Epsilon: probabilidade de explorar')
    validloss=[r for r in train if r['mean_loss']]
    b.plot([float(r['step'])/1000 for r in validloss],[float(r['mean_loss']) for r in validloss],color=CYAN,alpha=.6)
    b.set(xlabel='Passos (milhares)',ylabel='Erro de Bellman (Huber médio)')
    style(a);style(b);fig.tight_layout();fig.savefig(graphs/'diagnosticos.png',dpi=170);plt.close(fig)
    # PDF de 2 slides: é atualizado automaticamente para os resultados da pasta escolhida.
    with PdfPages(folder/'slides_resultados.pdf') as pdf:
        fig=plt.figure(figsize=(13.333,7.5),facecolor=BG)
        fig.text(.05,.92,'MountainCar: treinamento com DQN',fontsize=27,weight='bold')
        fig.text(.05,.86,'Rede MLP 2–64–64–3 com ReLU e recompensa original −1 por passo',fontsize=15,color=MUTED)
        ax=fig.add_axes([.37,.23,.58,.54]);validation_plot(ax,val,selected)
        info=(f"{execution['steps_completed']:,} passos\n".replace(',','.')+
              f"Seed de treino: {config['seed']}\n\nReplay: {config['buffer_size']:,}\n".replace(',','.')+
              f"Lote: {config['batch_size']}\nGamma: {config['gamma']}\nAdam: {config['learning_rate']}\n\nValidação sem exploração\n10 episódios por ponto")
        fig.text(.05,.75,info,fontsize=15,va='top',linespacing=1.5)
        fig.text(.05,.10,f'Modelo escolhido pela validação no passo {selected:,}.'.replace(',','.'),fontsize=14,color=CYAN)
        fig.text(.05,.045,'Gustavo Marques Oliveira / IA · UFU / Desafio 2',fontsize=10,color=MUTED)
        pdf.savefig(fig);fig.savefig(graphs/'slide_1.png',dpi=150);plt.close(fig)
        fig=plt.figure(figsize=(13.333,7.5),facecolor=BG)
        fig.text(.05,.92,'Agente treinado versus ações aleatórias',fontsize=27,weight='bold')
        n=summary['dqn']['episodes'];pr=summary['protocol']
        fig.text(.05,.86,f'{n} tentativas por agente, mesmas posições iniciais; teste sem atualizar os pesos.',fontsize=15,color=MUTED)
        a=fig.add_axes([.10,.30,.35,.44]);b=fig.add_axes([.58,.30,.35,.44]);compare_plot(a,b,summary)
        mean=summary['dqn']['mean_steps_success']
        success_text=(f'{mean:.2f}'.replace('.',',')+' passos em média nos sucessos.') if mean is not None else 'Não houve sucesso no teste.'
        fig.text(.05,.17,'DQN: '+success_text,fontsize=17,color=CYAN)
        fig.text(.05,.105,f'Teste: seeds {pr["seed_start"]}–{pr["seed_end"]}, separadas da validação. Um único treino.',fontsize=12,color=MUTED)
        fig.text(.05,.065,'Os resultados descrevem estas tentativas; não garantem sucesso em todas as execuções.',fontsize=12,color=MUTED)
        pdf.savefig(fig);fig.savefig(graphs/'slide_2.png',dpi=150);plt.close(fig)
    text=(f"RESULTADOS REAIS DA PASTA {folder}\nTreino: seed {config['seed']}, {execution['steps_completed']} passos.\n"
          f"Modelo selecionado na validação: passo {selected}.\n"
          f"Aleatório: {summary['aleatorio']['successes']}/{n} sucessos; retorno {summary['aleatorio']['mean_return']:.2f}.\n"
          f"DQN: {summary['dqn']['successes']}/{n} sucessos; retorno {summary['dqn']['mean_return']:.2f}.\n"
          f"Passos médios nos sucessos do DQN: {mean}.\n"
          "Não confundir retorno de treino com teste. As curvas podem oscilar.\n"
          "O teste usa 100 seeds por padrão, mas apenas um treinamento; não mede variação entre treinos.\n")
    (folder/'resultados_para_fala.txt').write_text(text,encoding='utf-8')
    print(text)
    print(f'Gráficos em {graphs}; apresentação em {folder / "slides_resultados.pdf"}')


if __name__ == '__main__':
    main()
