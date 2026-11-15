"""Mostra o agente salvo em uma janela; opcionalmente grava MP4 sem janela."""
import argparse
from pathlib import Path
import gymnasium as gym
import numpy as np
from dqn import ENV_ID, fix_seed, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo', default='runs/referencia/melhor_modelo.pt')
    p.add_argument('--modo', choices=['treinado', 'aleatorio'], default='treinado')
    p.add_argument('--episodios', type=int, default=3)
    p.add_argument('--seed', type=int, default=30_000)
    p.add_argument('--sem-janela', action='store_true')
    p.add_argument('--video', help='Caminho MP4. Grava sem abrir janela.')
    args = p.parse_args()
    if args.episodios <= 0 or args.seed < 0:
        p.error('Episódios devem ser positivos e seed não negativa.')
    fix_seed(123)
    rng = np.random.default_rng(12345)
    model = None
    if args.modo == 'treinado':
        model, checkpoint = load_model(args.modelo)
        print(f'Modelo treinado até o passo {checkpoint["step"]}. Sem aprendizagem durante a demonstração.')
    render_mode = 'rgb_array' if args.video else None if args.sem_janela else 'human'
    env = gym.make(ENV_ID, render_mode=render_mode)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        path = Path(args.video); path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=30, codec='libx264', macro_block_size=1)
    try:
        for ep in range(args.episodios):
            state, _ = env.reset(seed=args.seed + ep)
            total, steps = 0.0, 0
            if writer: writer.append_data(env.render())
            while True:
                action = int(rng.integers(3)) if model is None else model.action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward; steps += 1
                if writer: writer.append_data(env.render())
                if terminated or truncated:
                    result = 'SUCESSO' if terminated else 'LIMITE DE PASSOS'
                    print(f'Tentativa {ep+1} | seed {args.seed+ep} | {result} | passos={steps} | retorno={total:.0f}', flush=True)
                    break
    except KeyboardInterrupt:
        print('\nDemonstração encerrada pelo usuário.')
    finally:
        env.close()
        if writer: writer.close()


if __name__ == '__main__':
    main()
