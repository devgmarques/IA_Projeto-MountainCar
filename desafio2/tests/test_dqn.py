"""Verificações do alvo de Bellman, replay e salvamento/carregamento."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from dqn import QNetwork, ReplayBuffer, bellman_target, fix_seed, load_model, save_model

class DQNTests(unittest.TestCase):
    def test_goal_has_no_bootstrap_but_timeout_has(self):
        r = torch.tensor([-1.0, -1.0])
        # Caso 1: chegou. Caso 2: limite externo, mas não chegou.
        term = torch.tensor([1.0, 0.0])
        got = bellman_target(r, term, torch.tensor([-10.0, -10.0]), 0.99)
        torch.testing.assert_close(got, torch.tensor([-1.0, -10.9]))

    def test_replay_keeps_transitions_after_wrap(self):
        b = ReplayBuffer(2, 42)
        for k in range(3):
            b.add([k, 0], k, -1, [k+1, 0], k == 2)
        self.assertEqual(b.size, 2)
        s,a,r,ns,t = b.sample(2)
        self.assertEqual(set(a.tolist()), {1,2})
        torch.testing.assert_close(ns[:,0], s[:,0]+1)

    def test_saved_weights_reproduce_q_values(self):
        fix_seed(42)
        model = QNetwork()
        state = torch.tensor([[-0.5, 0.0], [0.2, 0.04]])
        self.assertEqual(tuple(model(state).shape), (2,3))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.pt'
            save_model(path, model, {'hidden':64}, 0, 0)
            loaded,_ = load_model(path)
            torch.testing.assert_close(model(state), loaded(state))

if __name__ == '__main__':
    unittest.main()
