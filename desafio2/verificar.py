"""Verifica versões e executa uma transição do ambiente, sem treinamento."""
import sys
import platform
import importlib
from pathlib import Path


def main():
    print('Python:', platform.python_version())
    print('Interpretador:', sys.executable)
    print('Pasta atual:', Path.cwd())
    if sys.version_info[:2] != (3, 12):
        print('AVISO: o pacote foi preparado para Python 3.12. Use o ambiente indicado no guia.')
    for name in ['torch', 'gymnasium', 'numpy', 'matplotlib', 'pygame', 'imageio', 'reportlab']:
        try:
            mod = importlib.import_module(name)
            print(name, getattr(mod, '__version__', 'OK'))
        except ImportError as e:
            raise SystemExit(f'Falta {name}: execute os comandos de instalação do guia.\n{e}')
    import gymnasium as gym
    from dqn import QNetwork, fix_seed
    fix_seed(42)
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=42)
    action = QNetwork().action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    assert state.shape == (2,) and reward == -1
    assert env.spec.max_episode_steps == 200
    env.close()
    print('OK: rede, ambiente e recompensa original funcionando. Este teste não treina o agente.')

if __name__ == '__main__':
    main()
