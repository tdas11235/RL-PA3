import numpy as np
import heapq

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
    def __init__(self, name, env, policy=None, fixer=2, alpha=0.1, gamma=0.99):
        self.name = name
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.grid_shape = (5, 5)
        self.walls = self._extract_walls()

    def _is_terminal(self, state):
        raise NotImplementedError

    def _is_valid(self, state):
        return True
    
    def _extract_walls(self):
        walls = set()
        desc = self.env.unwrapped.desc
        for r in range(5):
            for c in range(5):
                # Right wall
                if desc[r][2 * c + 2] == b"|":
                    walls.add(((r, c), (r, c + 1)))
                    walls.add(((r, c + 1), (r, c)))
                # Bottom wall
                if desc[r + 1][2 * c] == b"-":
                    walls.add(((r, c), (r + 1, c)))
                    walls.add(((r + 1, c), (r, c)))
        return walls
    
    def _get_neighbors(self, r, c):
        neighbors = []
        taxi_rowcol_to_state = lambda row, col: self.env.unwrapped.encode(row, col, 0, 0)
        state = taxi_rowcol_to_state(r, c)
        for action in range(4):  # Only consider movement actions: 0=South, 1=North, 2=East, 3=West
            transitions = self.env.P[state][action]
            for prob, next_state, reward, done in transitions:
                if prob > 0.0:
                    nr, nc, _, _ = decode_state(next_state)
                    neighbors.append(((nr, nc), action))
        return neighbors
    
    def _dijkstra(self, start, goal):
        heap = [(0, start, [])]
        visited = set()
        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            # list of actions
            if current == goal:
                return path
            for neighbor, action in self._get_neighbors(*current):
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + 1, neighbor, path + [action]))
        return []

    def execute(self, state):
        taxi_row, taxi_col, _, _ = decode_state(state)
        print(f"{self.name} invoked")
        path = self._dijkstra((taxi_row, taxi_col), self.target_location)
        total_reward = 0
        cum_rew = 0
        steps = 0
        done = False
        print("Start State", decode_state(state))
        print(path)
        for action in path:
            if done:
                break
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            done = terminated or truncated
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            cum_rew += reward
            steps += 1
        print(f"Reward: {cum_rew}, Steps: {steps}")
        print("End State", decode_state(state))
        return state, total_reward, steps, done, terminated, truncated, cum_rew
    

class GoToOption(BaseOption):
    def __init__(self, location_name, env, policy, alpha=0.1, gamma=0.99):
        super().__init__(f"go_to_{location_name}",
                        env, policy, alpha=alpha, gamma=gamma)
        self.target_location = LOCATION_COORDS[location_name]
        self.graph = self._build_graph()

    def _is_terminal(self, state):
        taxi_row, taxi_col, _, _ = decode_state(state)
        return (taxi_row, taxi_col) == self.target_location

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, _ = decode_state(state)
        return (taxi_row, taxi_col) != self.target_location
    
    def _build_graph(self):
        graph = {}
        for state in range(self.env.observation_space.n):
            taxi_row, taxi_col, _, _ = decode_state(state)
            pos = (taxi_row, taxi_col)
            if pos not in graph:
                graph[pos] = []
            for action in range(4):  # only movement actions
                transitions = self.env.unwrapped.P[state][action]
                for prob, next_state, _, _ in transitions:
                    if prob > 0:
                        next_row, next_col, _, _ = decode_state(next_state)
                        next_pos = (next_row, next_col)
                        if next_pos != pos:
                            graph[pos].append((next_pos, action))
        return graph

    def _dijkstra(self, start, goal):
        heap = [(0, start, [])]
        visited = set()
        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                return path
            for neighbor, action in self.graph.get(current, []):
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + 1, neighbor, path + [action]))
        return []

    def execute(self, state):
        taxi_row, taxi_col, _, _ = decode_state(state)
        # print(f"{self.name} invoked\nStart State {decode_state(state)}")
        path = self._dijkstra((taxi_row, taxi_col), self.target_location)
        # print(path)
        total_reward = 0
        cum_rew = 0
        steps = 0
        done = False
        for action in path:
            if done:
                break
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            cum_rew += reward
            steps += 1
        # print(f"Reward: {cum_rew}, Steps: {steps}")
        return state, total_reward, steps, done, terminated, truncated, cum_rew


class PickupOption(BaseOption):
    def __init__(self, env, policy):
        super().__init__("pickup", env, policy, fixer=0)
        self.action = 4

    def execute(self, state):
        # print(f"{self.name} invoked")
        next_state, reward, terminated, truncated, _ = self.env.step(self.action)
        done = terminated or truncated
        return next_state, reward, 1, done, terminated, truncated, reward

    def _is_terminal(self, state):
        _, _, pass_loc, _ = decode_state(state)
        return pass_loc == 4

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, _ = decode_state(state)
        return pass_loc < 4 and (taxi_row, taxi_col) == LOCATION_COORDS[LOCATION_NAMES[pass_loc]]


class DropoffOption(BaseOption):
    def __init__(self, env, policy):
        super().__init__("dropoff", env, policy, fixer=0)
        self.action = 5

    def execute(self, state):
        # print(f"{self.name} invoked")
        next_state, reward, terminated, truncated, _ = self.env.step(
            self.action)
        done = terminated or truncated
        return next_state, reward, 1, done, terminated, truncated, reward

    def _is_terminal(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        return pass_loc != 4 or (taxi_row, taxi_col) != LOCATION_COORDS[LOCATION_NAMES[dest]]

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        return pass_loc == 4 and (taxi_row, taxi_col) == LOCATION_COORDS[LOCATION_NAMES[dest]]
    

class SMDPAgent:
    def __init__(self, env, policy, opt_policies, alpha=0.1, gamma=0.99):
        self.env = env
        self.policy = policy
        self.opt_policies = opt_policies
        self.options = [
            GoToOption(loc, env, self.opt_policies[i], alpha, gamma) for i, loc in enumerate(LOCATION_NAMES)
        ] + [PickupOption(env, self.opt_policies[4]), DropoffOption(env, self.opt_policies[5])]
        self.q_table = np.zeros((env.observation_space.n, len(self.options)))
        self.alpha = alpha
        self.gamma = gamma

    def select_option(self, state):
        valid_options = [ind for ind, opt in enumerate(
            self.options) if opt._is_valid(state)]
        q_vals = self.q_table[state, valid_options]
        chosen_index = valid_options[self.policy.select_action(q_vals)]
        return chosen_index

    def train(self, n_episodes=500):
        rewards = []
        cumm_rewards = []
        for ep in range(n_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            cumm_reward = 0
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
                cumm_reward += cum_reward
                state = next_state
            rewards.append(total_reward)
            cumm_rewards.append(cumm_reward)
            self.policy.decay()
            if ep % 10 == 0:
                print(
                    f"Episode: {ep}, Reward: {cumm_reward}, Avg Reward: {np.mean(cumm_rewards[-100:])}")
                print(self.policy.epsilon)

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
                q_vals = self.q_table[state, valid_options]
                option = self.options[valid_options[np.argmax(q_vals)]]
                state, _, s, done, terminated, truncated, _ = option.execute(state)
                steps += s
            total_steps += steps
            success += terminated
        print(
            f"Success rate: {success/n_episodes:.4f}, Avg steps: {total_steps/n_episodes:.2f}")
