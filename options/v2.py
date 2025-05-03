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


# class BaseOption:
#     def __init__(self, name, env, policy, fixer=0, alpha=0.1, gamma=0.99):
#         self.name = name
#         self.env = env
#         self.policy = policy
#         self.q_table = np.zeros(
#             (env.observation_space.n, env.action_space.n - fixer))
#         self.alpha = alpha
#         self.gamma = gamma

#     def _is_terminal(self, state):
#         raise NotImplementedError

#     def _is_valid(self, state):
#         return True

#     def select_action(self, state):
#         return self.policy.select_action(self.q_table[state])

#     def learn(self, state, action, reward, next_state):
#         next_best = np.max(self.q_table[next_state])
#         td_target = reward + self.gamma * next_best
#         td_error = td_target - self.q_table[state, action]
#         self.q_table[state, action] += self.alpha * td_error

#     def execute(self, state):
#         total_reward = 0
#         steps = 0
#         done = False
#         cum_rew = 0
#         while not self._is_terminal(state) and not done:
#             action = self.select_action(state)
#             next_state, reward, terminated, truncated, _ = self.env.step(action)
#             done = terminated or truncated
#             self.learn(state, action, reward, next_state)
#             total_reward += (self.gamma ** steps) * reward
#             cum_rew += reward
#             steps += 1
#             state = next_state
#             self.policy.decay()
#             return state, total_reward, steps, done, terminated, truncated, cum_rew

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
        def taxi_rowcol_to_state(
            row, col): return self.env.unwrapped.encode(row, col, 0, 0)
        state = taxi_rowcol_to_state(r, c)
        # Only consider movement actions: 0=South, 1=North, 2=East, 3=West
        for action in range(4):
            transitions = self.env.P[state][action]
            for prob, next_state, reward, done in transitions:
                if prob > 0.0:
                    nr, nc, _, _ = decode_state(next_state)
                    neighbors.append(((nr, nc), action))
        return neighbors

class PickupOption(BaseOption):
    def __init__(self, location_name, env, policy, alpha=0.1, gamma=0.99):
        super().__init__(f"pickup_from_{location_name}",
                         env, policy, alpha=alpha, gamma=gamma)
        self.pickup_location = LOCATION_COORDS[location_name]
        self.graph = self._build_graph()

    def _is_terminal(self, state):
        taxi_row, taxi_col, pass_loc, _ = decode_state(state)
        # Terminal after successful pickup
        return pass_loc == 4  # 4 means passenger is in taxi

    def _is_valid(self, state):
        taxi_row, taxi_col, pass_loc, _ = decode_state(state)
        return pass_loc != 4

    def execute(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        path = self._dijkstra((taxi_row, taxi_col), self.pickup_location)

        total_reward = 0
        steps = 0
        done = False
        terminated = False
        truncated = False

        # First, navigate to pickup location
        for action in path:
            if done:
                break
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            done = terminated or truncated
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            steps += 1

        # Once at location, try to pickup
        if not done:
            pickup_action = 4  # fixed action for Pickup in Taxi-v3
            next_state, reward, terminated, truncated, _ = self.env.step(
                pickup_action)
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            steps += 1
            done = terminated or truncated

        return state, total_reward, steps, done, terminated, truncated, total_reward

    def _build_graph(self):
        # same as in GoToOption
        graph = {}
        for state in range(self.env.observation_space.n):
            taxi_row, taxi_col, _, _ = decode_state(state)
            pos = (taxi_row, taxi_col)
            if pos not in graph:
                graph[pos] = []
            for action in range(4):  # movement actions only
                transitions = self.env.unwrapped.P[state][action]
                for prob, next_state, _, _ in transitions:
                    if prob > 0:
                        next_row, next_col, _, _ = decode_state(next_state)
                        next_pos = (next_row, next_col)
                        if next_pos != pos:
                            graph[pos].append((next_pos, action))
        return graph

    def _dijkstra(self, start, goal):
        # same as in GoToOption
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


class DropoffOption(BaseOption):
    def __init__(self, location_name, env, policy, alpha=0.1, gamma=0.99):
        super().__init__(f"dropoff_at_{location_name}",
                         env, policy, alpha=alpha, gamma=gamma)
        self.dropoff_location = LOCATION_COORDS[location_name]
        self.graph = self._build_graph()

    def _is_terminal(self, state):
        _, _, pass_loc, _ = decode_state(state)
        # Terminal after successful dropoff (passenger disappears)
        return pass_loc != 4

    def _is_valid(self, state):
        _, _, pass_loc, _ = decode_state(state)
        return pass_loc == 4

    def execute(self, state):
        taxi_row, taxi_col, pass_loc, dest = decode_state(state)
        path = self._dijkstra((taxi_row, taxi_col), self.dropoff_location)

        total_reward = 0
        steps = 0
        done = False
        terminated = False
        truncated = False

        # Navigate to destination
        for action in path:
            if done:
                break
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            done = terminated or truncated
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            steps += 1

        # Drop off
        if not done:
            dropoff_action = 5  # fixed action for Dropoff in Taxi-v3
            next_state, reward, terminated, truncated, _ = self.env.step(
                dropoff_action)
            state = next_state
            total_reward += (self.gamma ** steps) * reward
            steps += 1
            done = terminated or truncated

        return state, total_reward, steps, done, terminated, truncated, total_reward

    def _build_graph(self):
        return PickupOption._build_graph(self)

    def _dijkstra(self, start, goal):
        return PickupOption._dijkstra(self, start, goal)