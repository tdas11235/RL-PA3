import numpy as np


# policy base class
class BasePolicy:
    def select_action(self, q_values: np.ndarray) -> int:
        raise NotImplementedError("Not overridden!")

    def decay(self) -> None:
        raise NotImplementedError("Not overridden!")
