output_file_name = 'all.py'
file_name = ['config', 'utils', 'env', 'buffer', 'agent', 'learning']

load_tensorboard = """
%load_ext tensorboard
%tensorboard --logdir logs
"""

libraries = """
import random
from collections import deque
import copy
import time
import math
import logging
import torch
from torch import nn
from torch import optim
from torch.functional import F
import numpy as np
import pickle
import matplotlib.pyplot as plt
\n\n
"""

write_drive = """
from google.colab import drive
drive.mount('/content/drive')
cfg.agent_path = f'/content/drive/My Drive/adjust_pose/agent.pkl'
\n\n
"""

# clear all.py
with open(f'src/{output_file_name}', 'w') as f:
    f.write('')

# load tensorboard and import all libraries 
with open(f'src/{output_file_name}', 'a') as f:
    f.write(load_tensorboard + libraries)

for p in file_name:
    with open(f'src/{p}.py', 'r') as f:
        # read line writen like '# cp config'
        lines = f.readlines()
        start = 0
        for line in lines:
            if f'# cp {p}' in line:
                start = lines.index(line)
                break
        end = len(lines)
    

        # copy the content from start to end
        with open(f'src/{output_file_name}', 'a') as f:
            f.writelines(lines[start:end] + ["\n\n"])
            if p == 'config':
                f.writelines('cfg = Config()\n')
                f.writelines(write_drive)
