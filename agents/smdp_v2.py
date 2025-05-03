from options.v2 import PickupOption, DropoffOption
import numpy as np


LOCATION_NAMES = ['R', 'G', 'Y', 'B']
LOCATION_COORDS = {
    'R': (0, 0),
    'G': (0, 4),
    'Y': (4, 0),
    'B': (4, 3)
}

class SMDPAgent:
    def __init__(self, env, policy, opt_policies, alpha=0.1, gamma=0.99):
        self.env = env
        self.policy = policy  # EpsilonGreedy or similar
        self.opt_policies = opt_policies  # 8 option policies (1 per option)
        self.alpha = alpha
        self.gamma = gamma
        self.options = [
            PickupOption(loc, env, opt_policies[i], alpha=alpha, gamma=gamma)
            for i, loc in enumerate(LOCATION_NAMES)
        ] + [
            DropoffOption(
                loc, env, opt_policies[i + 4], alpha=alpha, gamma=gamma)
            for i, loc in enumerate(LOCATION_NAMES)
        ]
        self.q_table = np.zeros((env.observation_space.n, len(self.options)))

    def select_option(self, state):
        # Only consider valid options
        valid_options = [i for i, opt in enumerate(
            self.options) if opt._is_valid(state)]
        if not valid_options:
            # If no valid options, pick a random one (shouldn't usually happen)
            return np.random.randint(len(self.options))
        q_vals = self.q_table[state, valid_options]
        action_idx = self.policy.select_action(q_vals)
        return valid_options[action_idx]

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

                # Execute the selected option
                next_state, reward, steps, done, terminated, truncated, cum_reward = option.execute(
                    state)

                # SMDP Q-learning update
                best_q_next = np.max(self.q_table[next_state])
                td_target = reward + (self.gamma ** steps) * best_q_next
                td_error = td_target - self.q_table[state, option_idx]
                self.q_table[state, option_idx] += self.alpha * td_error

                total_reward += reward
                state = next_state
            rewards.append(total_reward)
            avg_rewards.append(np.mean(rewards[-100:]))
            self.policy.decay()

            print(
                f"Episode: {ep+1}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_rewards[-1]:.2f}")

        return rewards, avg_rewards

    def play(self, n_episodes=10):
        success = 0
        total_steps = 0

        for ep in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0

            while not done:
                # Always pick the best valid option greedily
                valid_options = [i for i, opt in enumerate(
                    self.options) if opt._is_valid(state)]
                if not valid_options:
                    break
                q_vals = self.q_table[state, valid_options]
                best_option = self.options[valid_options[np.argmax(q_vals)]]

                # Execute best option
                state, _, s, done, terminated, truncated, _ = best_option.execute(
                    state)
                steps += s

            total_steps += steps
            success += int(terminated)

        print(
            f"Success rate: {success / n_episodes:.2f}, Avg steps: {total_steps / n_episodes:.1f}")
