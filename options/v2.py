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
        

class PickupOption(BaseOption):
    def __init__(self, env, target_color, policy, **kwargs):
        super().__init__(f"pickup_{target_color}", env, policy, **kwargs)
        self.target_color = target_color
        self.target_location = LOCATION_COORDS[target_color]

    def _is_terminal(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        # Ends when passenger is in taxi (pass_loc == 4)
        return pass_loc == 4

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        return pass_loc < 4 and LOCATION_NAMES[pass_loc] == self.target_color


class DropoffOption(BaseOption):
    def __init__(self, env, target_color, policy, **kwargs):
        super().__init__(f"dropoff_{target_color}", env, policy, **kwargs)
        self.target_color = target_color
        self.target_location = LOCATION_COORDS[target_color]

    def _is_terminal(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        # Ends when passenger is not in taxi and dropped at correct destination
        return pass_loc < 4 and LOCATION_NAMES[pass_loc] == LOCATION_NAMES[dest]

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        return pass_loc == 4 and LOCATION_NAMES[dest] == self.target_color


class SMDPAgent:
    def __init__(self, env, policy, opt_policies, alpha=0.1, gamma=0.99):
        self.env = env
        self.policy = policy
        self.opt_policies = opt_policies  # list of 8 policies: 4 pickup + 4 dropoff
        self.alpha = alpha
        self.gamma = gamma

        # Define 4 pickup and 4 dropoff options
        self.options = [
            PickupOption(env, color, opt_policies[i], alpha=alpha, gamma=gamma) for i, color in enumerate(LOCATION_NAMES)
        ] + [
            DropoffOption(env, color, opt_policies[i+4], alpha=alpha, gamma=gamma) for i, color in enumerate(LOCATION_NAMES)
        ]

        self.q_table = np.zeros((env.observation_space.n, len(self.options)))

    def select_option(self, state):
        valid_options = [i for i, opt in enumerate(
            self.options) if opt._is_valid(state)]
        if not valid_options:
            # fallback if no option is valid
            return np.random.randint(len(self.options))
        q_vals = self.q_table[state, valid_options]
        chosen_index = valid_options[self.policy.select_action(q_vals)]
        return chosen_index

    def train(self, n_episodes=500):
        rewards = []
        avg_rewards = []
        for ep in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                option_idx = self.select_option(state)
                option = self.options[option_idx]
                next_state, reward, steps, done, _, _, cum_reward = option.execute(
                    state)
                best_next = np.max(self.q_table[next_state])
                td_target = reward + (self.gamma ** steps) * best_next
                td_error = td_target - self.q_table[state, option_idx]
                self.q_table[state, option_idx] += self.alpha * td_error
                total_reward += reward
                state = next_state
            rewards.append(cum_reward)
            self.policy.decay()
            avg_rewards.append(np.mean(rewards[-100:]))
            print(
                f"Episode: {ep}, Reward: {total_reward}, Avg Reward: {avg_rewards[-1]:.2f}")
        return rewards, avg_rewards

    def play(self, n_episodes=10):
        success = 0
        total_steps = 0
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            while not done:
                valid_options = [i for i, opt in enumerate(
                    self.options) if opt._is_valid(state)]
                if not valid_options:
                    break
                q_vals = self.q_table[state, valid_options]
                option = self.options[valid_options[np.argmax(q_vals)]]
                state, _, s, done, terminated, truncated, _ = option.execute(
                    state)
                steps += s
            total_steps += steps
            success += terminated
        print(
            f"Success rate: {success/n_episodes:.4f}, Avg steps: {total_steps/n_episodes:.2f}")
