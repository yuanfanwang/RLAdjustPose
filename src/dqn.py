from env import Env
from agent import DQNAgent
from config import Config as cfg
from utils import plot_total_reward, save_obj

env = Env()
agent = DQNAgent()
reward_history = []

# next: reward clipping, double dqn, prioritized experience replay, dueling dqn
for episode in range(cfg.episodes+1):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info, _ = env.step(action)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    cync_interval = 20
    if episode % cync_interval == 0:
        agent.sync_target()
    
    print(f"episode: {episode}, total_reward: {total_reward}, time: {info['time']}")
    
    reward_history.append(total_reward)

save_obj(agent, cfg.agent_path)
plot_total_reward(reward_history)
