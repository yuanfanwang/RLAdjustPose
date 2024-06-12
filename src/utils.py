import pickle
import matplotlib.pyplot as plt


# cp utils

def plot_total_reward(reward_history):
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

def save_obj(obj, path):
    if hasattr(obj, 'writer'):
        obj.writer = None
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def create_writer():
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    from zoneinfo import ZoneInfo

    jst = ZoneInfo("Asia/Tokyo")
    current_time = datetime.datetime.now(jst).strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=f"logs/{current_time}")

    return writer
