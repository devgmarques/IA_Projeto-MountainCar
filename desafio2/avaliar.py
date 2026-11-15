"""Compara o DQN e ações aleatórias em condições iniciais pareadas, sem aprender."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import gymnasium as gym
import numpy as np
from dqn import ENV_ID, fix_seed, load_model


def summarize(rows):
    returns = np.array([r['return'] for r in rows], dtype=float)
    successes = [r['length'] for r in rows if r['success']]
    n, k = len(rows), len(successes)
    # IC de Wilson: incerteza da proporção para este modelo, não entre treinos.
    z = 1.96
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    margin = z * ((p*(1-p)/n + z*z/(4*n*n))**0.5) / (1+z*z/n)
    return dict(episodes=n, successes=k, success_rate=p,
                success_ci95=[max(0.0, center-margin), min(1.0, center+margin)],
                mean_return=float(returns.mean()),
                std_return=float(returns.std(ddof=1)) if n > 1 else 0.0,
                mean_steps_all=float(np.mean([r['length'] for r in rows])),
                mean_steps_success=float(np.mean(successes)) if successes else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pasta', default='runs/referencia')
    p.add_argument('--episodios', type=int, default=100)
    p.add_argument('--seed-inicial', type=int, default=20_000)
    args = p.parse_args()
    if args.episodios <= 0 or args.seed_inicial < 0:
        p.error('Use episódios positivos e seed inicial não negativa.')
    folder = Path(args.pasta)
    path = folder / 'melhor_modelo.pt'
    fix_seed(123)
    model, checkpoint = load_model(path)
    seeds = list(range(args.seed_inicial, args.seed_inicial + args.episodios))
    if set(seeds) & set(checkpoint['config']['validation_seeds']):
        p.error('Teste e validação devem usar seeds diferentes.')
    if checkpoint['config']['seed'] in seeds:
        p.error('Não use a seed inicial do treino no teste.')
    env = gym.make(ENV_ID)
    records, summary = [], {}
    for name in ['aleatorio', 'dqn']:
        rng = np.random.default_rng(12345)
        rows = []
        for episode, seed in enumerate(seeds, 1):
            state, _ = env.reset(seed=seed)
            initial_x = float(state[0])
            total, length = 0.0, 0
            while True:
                action = int(rng.integers(3)) if name == 'aleatorio' else model.action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward; length += 1
                if terminated or truncated:
                    rows.append(dict(agent=name, episode=episode, seed=seed, initial_position=initial_x,
                                     return_=total, length=length, success=int(terminated), truncated=int(truncated)))
                    rows[-1]['return'] = rows[-1].pop('return_')
                    break
        summary[name] = summarize(rows)
        records.extend(rows)
        print(f'{name}: sucesso {summary[name]["success_rate"]:.1%}; retorno médio {summary[name]["mean_return"]:.2f}')
    env.close()
    with (folder / 'avaliacao.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary['protocol'] = dict(environment=ENV_ID, seed_start=seeds[0], seed_end=seeds[-1],
                               action_seed=12345, deterministic_dqn=True, training_updates=False,
                               model_step=checkpoint['step'], model_episode=checkpoint['episode'],
                               model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               evaluated_utc=datetime.now(timezone.utc).isoformat(),
                               note='Uma execução de treinamento. Teste separado da seleção do modelo por validação.')
    (folder / 'resumo.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
