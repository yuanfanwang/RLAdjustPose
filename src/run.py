from env import Env
from config import Config as cfg
from utils import load_obj
from agent import ACAgent
from agent import ActorNet
from agent import CriticNet


env = Env()
agent = load_obj(cfg.agent_path)
agent.train = False

cfg.render = True
for _ in range(10):
    state, _ = env.reset()
    done = False
    truncate = False
    total_reward = 0

    while not done and not truncate:
        action = agent.get_action(state)
        next_state, reward, done, truncate, info, _ = env.step(action)
        state = next_state
        total_reward += reward

    print(f"total_reward: {total_reward}, time: {info['time']}")
