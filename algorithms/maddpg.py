import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork
from gym.spaces import Box, Discrete
from utils.misc import soft_update, hard_update, average_gradient, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent


MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    def __init__(self):
        pass