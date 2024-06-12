import numpy as np
import random
from collections import deque


# cp buffer

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if buffer_size < batch_size:
            raise ValueError("buffer_size should be larger than batch_size")

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state      = np.stack([x[0] for x in data])
        action     = np.stack([x[1] for x in data])
        reward     = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done       = np.stack([x[4] for x in data])
        # print(state, action, reward, next_state, done)
        return state, action, reward, next_state, done


class ReplayPrioritizedBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if buffer_size < batch_size:
            raise ValueError("buffer_size should be larger than batch_size")

    def add(self, state, action, reward, next_state, done, delta=1):
        data = (state, action, reward, next_state, done, delta)
        self.buffer.append(data)
    
    def __len__(self):
        return len(self.buffer)
    
    def get_batch(self):
        data = random.choices(self.buffer, weights=[x[5] for x in self.buffer], k=self.batch_size)
        state      = np.stack([x[0] for x in data])
        action     = np.stack([x[1] for x in data])
        reward     = np.stack([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done       = np.stack([x[4] for x in data])
        return state, action, reward, next_state, done
