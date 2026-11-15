import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from dqn import ENV_ID, QNetwork, ReplayBuffer, fix_seed, learn, save_model


def validate(model, seeds):
    env = gym.make(ENV_ID)
    returns, successes = [], []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        total = 0.0
        while True:
            state, reward, terminated, truncated, _ = env.step(model.action(state))
            total += reward
            if terminated or truncated:
                returns.append(total)
                successes.append(int(terminated))
                break
    env.close()
    return float(np.mean(returns)), float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', default='runs/meu_treino')
    parser.add_argument('--passos', type=int, default=250_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.passos <= 0:
        parser.error('--passos deve ser positivo')
    out = Path(args.saida)
    if out.exists() and any(out.iterdir()):
        parser.error('A pasta de saída já contém arquivos. Use outro nome para preservar os resultados.')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(env=ENV_ID, steps=args.passos, seed=args.seed, hidden=64,
                  gamma=0.99, learning_rate=0.001, batch_size=64, buffer_size=50_000,
                  learning_starts=2_000, train_every=4, target_every=1_000,
                  epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=75_000,
                  eval_every=10_000, validation_seeds=list(range(10_000, 10_010)),
                  max_episode_steps=200, reward_shaping=False, device='cpu')
    if args.seed < 0:
        parser.error('seed deve ser não negativa')
    fix_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = gym.make(ENV_ID)
    online, target = QNetwork(config['hidden']), QNetwork(config['hidden'])
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config['learning_rate'])
    replay = ReplayBuffer(config['buffer_size'], args.seed + 1)
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), platform=platform.platform(),
                    torch=str(torch.__version__), gymnasium=gym.__version__, numpy=np.__version__,
                    processor=platform.processor(), origin='executed training; see config.json')
    start = time.perf_counter()
    best_score = -float('inf')
    episode, step = 0, 0
    state, _ = env.reset(seed=args.seed)
    ep_return, ep_steps, losses = 0.0, 0, []
    interrupted = False
    with (out / 'treino.csv').open('w', newline='', encoding='utf-8') as ft, \
         (out / 'validacao.csv').open('w', newline='', encoding='utf-8') as fv:
        train_csv, val_csv = csv.writer(ft), csv.writer(fv)
        train_csv.writerow(['episode', 'step', 'return', 'length', 'success', 'epsilon', 'mean_loss'])
        val_csv.writerow(['step', 'mean_return', 'success_rate'])
        def evaluate(step):
            nonlocal best_score
            score, success = validate(online, config['validation_seeds'])
            val_csv.writerow([step, score, success]); fv.flush()
            if score > best_score:
                best_score = score
                save_model(out / 'melhor_modelo.pt', online, config, step, episode,
                           dict(mean_return=score, success_rate=success))
            print(f'VALIDAÇÃO passo={step:,} retorno={score:.1f} sucesso={success:.0%} melhor={best_score:.1f}', flush=True)
        evaluate(0)
        try:
            for step in range(1, args.passos + 1):
                progress = min(1.0, (step - 1) / config['epsilon_decay_steps'])
                epsilon = config['epsilon_start'] + progress * (config['epsilon_end'] - config['epsilon_start'])
                action = int(rng.integers(3)) if rng.random() < epsilon else online.action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                replay.add(state, action, reward, next_state, terminated)
                state = next_state
                ep_return += reward; ep_steps += 1
                if step >= config['learning_starts'] and step % config['train_every'] == 0:
                    losses.append(learn(online, target, optimizer, replay, config['batch_size'], config['gamma']))
                if step % config['target_every'] == 0:
                    target.load_state_dict(online.state_dict())
                if terminated or truncated:
                    episode += 1
                    mean_loss = float(np.mean(losses)) if losses else ''
                    train_csv.writerow([episode, step, ep_return, ep_steps, int(terminated), epsilon, mean_loss])
                    ft.flush()
                    if episode % 50 == 0:
                        print(f'Episódio {episode} | passo {step:,} | retorno {ep_return:.0f} | epsilon {epsilon:.3f} | {time.perf_counter()-start:.0f}s', flush=True)
                    # Fluxo de seeds separado das sementes fixas de validação/teste.
                    state, _ = env.reset()
                    ep_return, ep_steps, losses = 0.0, 0, []
                if step % config['eval_every'] == 0:
                    evaluate(step)
                    save_model(out / 'ultimo_modelo.pt', online, config, step, episode)
        except KeyboardInterrupt:
            interrupted = True
            print('\nInterrompido: salvando os pesos disponíveis.', flush=True)
        finally:
            save_model(out / 'ultimo_modelo.pt', online, config, step, episode)
            if not interrupted and step % config['eval_every'] != 0:
                evaluate(step)
            env.close()
    metadata.update(elapsed_seconds=time.perf_counter()-start, steps_completed=step,
                    episodes_completed=episode, interrupted=interrupted,
                    best_validation_return=best_score,
                    partial_episode_steps=ep_steps)
    (out / 'execucao.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Concluído. Modelo para demonstrar: {out / "melhor_modelo.pt"}', flush=True)


if __name__ == '__main__':
    main()
