import numpy as np


# cp config

class Config:
    """
    change the 'mode' and 'actions' to switch the task
    """
    mode = "x"
    agent_type = "cac"
    # actions = {0: [0, 0], 1: [1, 0], 2: [-1, 0]}  # vx 1
    actions = {0: [0, 0], 1: [1, 0], 2: [-1, 0], 3: [2, 0], 4: [-2, 0]}  # vx 2
    # actions = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1], 7: [1, 0], 8: [1, 1]}  # vx1, wz1

    # static
    render = False
    episodes = 10000
    buffer_size = 10000
    batch_size = 32
    fps = 50
    resolution = np.array([0.001, 0.001, 0.1 * 3.14 / 180])
    agent_path = "data/rl-continuous-x-adjust/agent.pkl"
    time_limit = 20
    state_size = 5
    sync_interval = 20
