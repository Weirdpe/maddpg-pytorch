import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork
from gym.spaces import Box, Discrete
from utils.misc import soft_update, hard_update, average_gradient, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent


MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    def __init__(self, agent_init_params, alg_type,
                 gamma=0.95,tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        self.nagents = len(alg_type)
        self.alg_type = alg_type
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim, **params) for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0



