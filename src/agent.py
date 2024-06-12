import copy
import torch
from torch import nn
from torch import optim
from torch.functional import F
from torch.distributions import Normal
import numpy as np
from config import Config as cfg
from buffer import  ReplayPrioritizedBuffer


# cp agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

class ActorNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        if cfg.agent_type == "dac":
            x = torch.softmax(x, dim=1)
        return x

class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        return x

class ACAgent:
    def __init__(self):
        self.gamma = 0.99

        ##### minus dist reward
        # x adjust (batch=32, input=5, hidden=128, output=5, gamma=0.9999, Even if it were 0.98, the result would be the same)
        # self.lr_pi = 5e-6 # coverage slower than 2e-5 but not NaN, and will get better value (20240513-223456 vs 20240513-205313)
        # self.lr_v  = 5e-5 # coverage slower than 5e-5 but not NaN, and will get better value (20240513-223456 vs 20240513-205313)

        ##### plus dist reward
        ##### xa adjust (commit: 9ad0ef49e65d28c4e3fd00d5776a3c0b3ffad9a2)
        self.lr_pi = 2e-4
        self.lr_v  = 5e-4

        self.state_size = 5

        self.hidden_size = 128

        if cfg.agent_type == "dac":
            self.action_size = len(cfg.actions)
        if cfg.agent_type == "cac":
            if cfg.mode == "x":
                self.action_size = 2
            elif cfg.mode == "xa":
                self.action_size = 4

        self.pi = ActorNet(self.state_size, self.hidden_size, self.action_size).to(device)
        self.v = CriticNet(self.state_size, self.hidden_size, 1).to(device)
        self.v_target = CriticNet(self.state_size, self.hidden_size, 1).to(device)
        self.optimizer_pi = optim.AdamW(self.pi.parameters(), self.lr_pi)
        self.optimizer_v = optim.AdamW(self.v.parameters(), self.lr_v)

        self.writer = None
        self.pi_l1_hook = None
        self.pi_l2_hook = None
        self.v_l1_hook = None
        self.v_l2_hook = None

        self.episode = 0
        self.update_per_episode = 0
        self.train = True
    
    def sync_v_target(self):
        self.v_l1_hook.remove()
        self.v_l2_hook.remove()
        self.v_target = copy.deepcopy(self.v)
        self.v_l1_hook = self.v.l1.register_forward_hook(self.value1_hook)
        self.v_l2_hook = self.v.l2.register_forward_hook(self.value2_hook)

    def value1_hook(self, model, input, output):
        if not self.train: return
        if self.update_per_episode % 100 == 0:
            self.writer.add_histogram('ValueNet/l1 activation', output.detach().cpu().numpy(), self.episode)

    def value2_hook(self, model, input, output):
        if not self.train: return
        if self.update_per_episode % 100 == 0:
            self.writer.add_histogram('ValueNet/l2 activation', output.detach().cpu().numpy(), self.episode)

    def policy1_hook(self, model, input, output):
        if not self.train: return
        if self.update_per_episode % 100 == 0:
            self.writer.add_histogram('PolicyNet/l1 activation', output.detach().cpu().numpy(), self.episode)

    def policy2_hook(self, model, input, output):
        if not self.train: return
        if self.update_per_episode % 100 == 0:
            self.writer.add_histogram('PolicyNet/l2 activation', output.detach().cpu().numpy(), self.episode)
   
    def set_writer(self, writer):
        self.writer = writer
        self.v_l1_hook = self.v.l1.register_forward_hook(self.value1_hook)    
        self.v_l2_hook = self.v.l2.register_forward_hook(self.value2_hook)    
        self.pi_l1_hook = self.pi.l1.register_forward_hook(self.policy1_hook)
        self.pi_l2_hook = self.pi.l2.register_forward_hook(self.policy2_hook)
    
    def print_value_graph(self):
        X = [i for i in range(-10, 11)]
        for x in X:
            state = torch.tensor([x*0.02, 0, 0, 0, 0], dtype=torch.float32).to(device)
            v = self.v(state).detach().cpu().numpy()
            print(f'x: {x*0.02: .2f}  v: {v[0]: .2f}')
    
    def set_episode(self, episode):
        self.episode = episode
        self.update_per_episode = 0

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)

        if cfg.agent_type == "dac":
            probs = self.pi(state)[0]
            if self.train:
                action = np.random.choice(self.action_size, p=probs.detach().cpu().numpy())
            else:
                action = torch.argmax(probs).item()
            return action
        elif cfg.agent_type == "cac":
            if cfg.mode == "x":
                av, log_std = self.pi(state)[0].detach()
                if self.train:
                    std = torch.exp(log_std)
                    normal_dist = Normal(av, std)
                    action = normal_dist.sample()
                    return action.item()
                else:
                    return av.item()
            elif cfg.mode == "xa":
                pass
    
    def get_loss(self, state, action, reward, next_state, done):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state[np.newaxis, :], dtype=torch.float32).to(device)

        target = reward + self.gamma * self.v_target(next_state).detach().cpu().numpy().squeeze() * (1 - done)
        target = torch.tensor(target, dtype=torch.float32)
        v = self.v_target(state).detach().cpu().squeeze()

        delta = target - v
        qs = self.pi(state).detach().cpu()
        action_prob = qs[0, action]
        loss_pi = -torch.log(action_prob) * delta
        loss_pi = float(abs(loss_pi))
        return loss_pi


    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        if cfg.agent_type == "cac":
            action = torch.tensor(action, dtype=torch.float32).to(device).unsqueeze(0)
        
        target = reward + self.gamma * self.v_target(next_state).detach().cpu().numpy().squeeze() * (1 - done)
        target = torch.tensor(target, dtype=torch.float32).to(device).unsqueeze(1)
        v = self.v(state)
        loss_v = F.mse_loss(v, target)

        delta = reward + self.gamma * self.v_target(next_state).detach().cpu().numpy().squeeze() * (1 - done)- self.v_target(state).detach().cpu().numpy().squeeze()
        delta = torch.tensor(delta, dtype=torch.float32).to(device).unsqueeze(1)
        qs = self.pi(state)

        if cfg.agent_type == "dac":
            action_prob = qs[np.arange(len(action)), action].unsqueeze(1)
            loss_pi = torch.mean(-torch.log(action_prob) * delta)
        elif cfg.agent_type == "cac":
            if cfg.mode == "x":
                av, log_std = qs[:, 0], qs[:, 1]
                std = torch.exp(log_std)
                normal_dist = Normal(av, std)
                action_log_prob = normal_dist.log_prob(action)
                loss_pi = torch.mean(-action_log_prob * delta)
            elif cfg.mode == "xa":
                pass

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()

        self.optimizer_v.step()
        self.optimizer_pi.step()

        if self.update_per_episode % 5 == 0:
            v_l1_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.v.parameters() if p.grad is not None]), 2)
            v_l2_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in self.v.parameters() if p.grad is not None]), 2)
            pi_l1_grad_norm = torch.norm(self.pi.l1.weight.grad.detach(), 2)
            pi_l2_grad_norm = torch.norm(self.pi.l2.weight.grad.detach(), 2)
            self.writer.add_scalar('Gradient Policy l1', pi_l1_grad_norm.item(), self.episode)
            self.writer.add_scalar('Gradient Policy l2', pi_l2_grad_norm.item(), self.episode)
            self.writer.add_scalar('Gradient Value l1', v_l1_grad_norm.item(), self.episode)
            self.writer.add_scalar('Gradient Value l2', v_l2_grad_norm.item(), self.episode)
            self.writer.add_scalar('Loss Policy', loss_pi.item(), self.episode)
            self.writer.add_scalar('Loss Value', loss_v.item(), self.episode)

        self.update_per_episode += 1


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(cfg.state_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x

# change gamma, lr, epsilon, buffer_size, batch_size, episode, sync_interval and layer structure to fit the task
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 1e-4 * 5
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = len(cfg.actions)

        self.replay_buffer = ReplayPrioritizedBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size).to(device)
        self.qnet_target = QNet(self.action_size).to(device)
        self.optimizer = optim.AdamW(self.qnet.parameters(), self.lr)
        self.loss_fn = nn.MSELoss()

    def sync_target(self):
        self.qnet_target = copy.deepcopy(self.qnet)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)
            qs = self.qnet(state)
            return torch.argmax(qs).item()

    def update(self, state, action, reward, next_state, done):
        # for prioritized experience replay
        target = reward + self.gamma * self.qnet_target(
            torch.tensor(next_state[np.newaxis, :],
                         dtype=torch.float32).to(device)).max().item() * (1 - done)
        q = self.qnet(torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device))[0, action].detach()
        delta = abs(target - q)
        self.replay_buffer.add(state, action, reward, next_state, done, delta)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        qs = self.qnet(state)
        # qs の 各行に対して、action に対応する列の値を取り出す
        q = qs[np.arange(self.batch_size), action]
        
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1).values.detach().numpy()
        target = reward + self.gamma * next_q * (1 - done)

        loss = self.loss_fn(q, torch.tensor(target, dtype=torch.float32).to(device))
        self.qnet.zero_grad()
        self.loss_fn.zero_grad()
        loss.backward()
        self.optimizer.step()
