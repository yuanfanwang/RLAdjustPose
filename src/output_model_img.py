from agent import ACAgent
import torch
from torchviz import make_dot

from torch import nn

agent = ACAgent()
image = make_dot(agent.pi(torch.rand(1, 5)), params=dict(agent.pi.named_parameters()))
image.format = 'png'
image.render("pi", directory="model_image/")

image = make_dot(agent.v(torch.rand(1, 5)), params=dict(agent.v.named_parameters()))
image.format = 'png'
image.render("v", directory="model_image/")
