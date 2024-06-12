from env import Env
from agent import ACAgent
from buffer import ReplayBuffer, ReplayPrioritizedBuffer
from config import Config as cfg
from utils import save_obj, create_writer

# cp learning

env = Env()
# buffer = ReplayPrioritizedBuffer(cfg.buffer_size, cfg.batch_size)
buffer = ReplayBuffer(cfg.buffer_size, cfg.batch_size)
agent = ACAgent()
writer = create_writer()
agent.set_writer(writer)

try:
    for episode in range(cfg.episodes+1):
        state, _ = env.reset()
        done = False
        truncate = False
        total_reward = 0
        agent.set_episode(episode)
        while (not done) and (not truncate):
            # move
            action = agent.get_action(state)
            next_state, reward, done, truncate, info, _ = env.step(action)

            # buffering
            # loss = agent.get_loss(state, action, reward, next_state, done)
            # buffer.add(state, action, reward, next_state, done, loss)
            buffer.add(state, action, reward, next_state, done)

            pause = 2 * (cfg.time_limit * cfg.fps)
            # update
            if len(buffer) > pause:
                agent.update(*buffer.get_batch())

            # next
            state = next_state
            total_reward += reward

        if episode % cfg.sync_interval == 0:
            agent.sync_v_target()

        if episode % 100 == 0:
            agent.print_value_graph()

        print(f"ep: {episode}, total_reward: {total_reward}, time: {info['time']}")
        writer.add_scalar('Reward/Total', total_reward, episode)
except KeyboardInterrupt:
    save_obj(agent, cfg.agent_path)
    print("agent saved")

save_obj(agent, cfg.agent_path)
