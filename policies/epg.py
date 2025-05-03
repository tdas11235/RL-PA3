import numpy as np
from policies.base import BasePolicy

# epslion-greedy policy class


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 1.0, min_epsilon: float = 0.1, decay_factor: float = 0.99) -> None:
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor

    def select_action(self, q_values: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(q_values))
        return int(np.argmax(q_values))

    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)
