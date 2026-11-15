"""DQN para MountainCar: rede, replay buffer e atualização de Bellman.
Implementação didática em PyTorch, sem biblioteca pronta de agentes de RL.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

ENV_ID = 'MountainCar-v0'


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # Rede pequena: evita overhead de muitas threads.
    torch.use_deterministic_algorithms(True)


class QNetwork(nn.Module):
    """2 entradas -> 64 ReLU -> 64 ReLU -> 3 valores Q (sem softmax)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.register_buffer('low', torch.tensor([-1.2, -0.07]))
        self.register_buffer('high', torch.tensor([0.6, 0.07]))
        self.layers = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, state):
        # Escalas comparáveis para posição e velocidade, sem usar dados futuros.
        normalized = 2.0 * (state - self.low) / (self.high - self.low) - 1.0
        return self.layers(normalized)

    @torch.no_grad()
    def action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return int(self(state).argmax(dim=1).item())


class ReplayBuffer:
    """Memória circular de experiências (s, a, r, s', terminated)."""
    def __init__(self, capacity, seed):
        self.states = np.zeros((capacity, 2), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.index = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, terminated):
        i = self.index
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.from_numpy(array[indices]) for array in
                     (self.states, self.actions, self.rewards, self.next_states, self.terminated))


def bellman_target(rewards, terminated, next_q, gamma):
    # Somente a chegada à meta elimina o valor futuro.
    # O limite externo de 200 passos (truncated) NÃO é um terminal físico.
    return rewards + gamma * (1.0 - terminated) * next_q


def learn(online, target, optimizer, replay, batch_size, gamma):
    states, actions, rewards, next_states, terminated = replay.sample(batch_size)
    predicted = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states).max(dim=1).values
        expected = bellman_target(rewards, terminated, next_q, gamma)
    loss = nn.functional.smooth_l1_loss(predicted, expected)  # Huber
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def save_model(path, model, config, step, episode, validation=None):
    torch.save({'state_dict': model.state_dict(), 'config': config,
                'step': step, 'episode': episode, 'validation': validation,
                'algorithm': 'DQN', 'reward': 'original -1 per step'}, path)


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Modelo não encontrado: {path}. Confira --modelo ou execute treinar.py.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = QNetwork(checkpoint['config']['hidden'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
