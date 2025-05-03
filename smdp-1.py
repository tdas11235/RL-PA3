import gymnasium as gym
from agents.smdp_dijkstra import SMDPAgent
from policies.epg import EpsilonGreedyPolicy
from policies.sm import SoftmaxPolicy

env = gym.make("Taxi-v3")
policy = EpsilonGreedyPolicy(epsilon=1, decay_factor=0.995, min_epsilon=0.0005)
opt_policies = [
    None
    for _ in range(6)
]

agent = SMDPAgent(env, policy, opt_policies, gamma=0.99, alpha=0.01)
agent.train(n_episodes=7500)
agent.play(n_episodes=100)
