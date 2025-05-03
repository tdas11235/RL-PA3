import numpy as np

LOCATION_NAMES = ['R', 'G', 'Y', 'B']
LOCATION_COORDS = {
    'R': (0, 0),
    'G': (0, 4),
    'Y': (4, 0),
    'B': (4, 3)
}


def decode_state(state):
    taxi_row, taxi_col, pass_loc, dest = list(
        np.unravel_index(state, [5, 5, 5, 4]))
    return taxi_row, taxi_col, pass_loc, dest


class BaseOption:
  def __init__(self, name, env, policy, fixer=2, alpha=0.1, gamma=0.99):
    self.name = name
    self.env = env
    self.policy = policy
    self.q_table = np.zeros(
        (env.observation_space.n, env.action_space.n - fixer))
    self.alpha = alpha
    self.gamma = gamma

  def _is_terminal(self, state):
    raise NotImplementedError

  def _is_valid(self, state):
    return True

  def select_action(self, state):
    return self.policy.select_action(self.q_table[state])

  def learn(self, state, action, reward, next_state):
    next_best = np.max(self.q_table[next_state])
    td_target = reward + self.gamma * next_best
    td_error = td_target - self.q_table[state, action]
    self.q_table[state, action] += self.alpha * td_error

  def execute(self, state):
    total_reward = 0
    steps = 0
    done = False
    cum_rew = 0
    while not self._is_terminal(state) and not done:
      action = self.select_action(state)
      next_state, reward, terminated, truncated, _ = self.env.step(action)
      done = terminated or truncated
      taxi_row, taxi_col, pass_loc, dest = decode_state(next_state)
      if hasattr(self, 'target_location'):
        goal_match = self.target_location == (
            LOCATION_COORDS[LOCATION_NAMES[pass_loc]
                            ] if pass_loc < 4 else LOCATION_COORDS[LOCATION_NAMES[dest]]
        )
        if (taxi_row, taxi_col) == self.target_location and goal_match:
            reward += 20
      self.learn(state, action, reward, next_state)
      total_reward += (self.gamma ** steps) * reward
      cum_rew += reward
      steps += 1
      state = next_state
    self.policy.decay()
    return state, total_reward, steps, done, terminated, truncated, cum_rew
  

