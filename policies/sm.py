import numpy as np
from policies.base import BasePolicy

class SoftmaxPolicy(BasePolicy):
    def __init__(self, tau: float = 1000.0, min_tau: float = 0.1, decay_factor: float = 0.9995, seed: int = 0) -> None:
        self.tau = tau
        self.min_tau = min_tau
        self.decay_factor = decay_factor

    def select_action(self, q_values: np.ndarray) -> int:
        q_values = np.array(q_values, dtype=np.float64)
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / self.tau)
        probabilities = exp_q / np.sum(exp_q)
        return int(np.random.choice(len(q_values), p=probabilities))

    def decay(self):
        self.tau = max(self.min_tau, self.tau * self.decay_factor)
