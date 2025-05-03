import gymnasium as gym
from agents.smdp_v2 import SMDPAgent
from policies.epg import EpsilonGreedyPolicy

env = gym.make("Taxi-v3")
policy = EpsilonGreedyPolicy(epsilon=0.5, decay_factor=0.99, min_epsilon=0.01)
opt_policies = [
    EpsilonGreedyPolicy(epsilon=0.1, decay_factor=0.99, min_epsilon=0.01)
    for _ in range(8)
]

agent = SMDPAgent(env, policy, opt_policies, gamma=0.99, alpha=0.01)
agent.train(n_episodes=7500)
agent.play(n_episodes=100)
