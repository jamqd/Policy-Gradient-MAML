import math
import torch
from torch import nn
from torch.distributions import Normal, Categorical

def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class Policy(nn.Module):

    def __init__(self, input_size, output_size, hidden_dims=[128, 128]):
        super(Policy, self).__init__()
        
        layers = [linear_init(nn.Linear(input_size, hidden_dims[0])), nn.ReLU()]
        for i, o in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hidden_dims[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        # std not a function of observation
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(1e-6)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action