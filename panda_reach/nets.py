import math


import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)



class StochasticActor(nn.Module):
    def __init__(
        self,
        log_std_low=-1.0,
        log_std_high=1.0,
        state_space_size=10,
        action_space_size=8,
        hidden_size=1024,
    ):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_space_size, hidden_size),

                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),

                                 nn.ReLU())

        #self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),

                                 #nn.ReLU())
        self.fc3 = nn.Linear(hidden_size, 2 * action_space_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        #x = self.fc3(x)
        out=self.fc3(x)
        mu, log_std = out.chunk(2, dim=1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
        ) * (log_std + 1)
        std = log_std.exp()
        dist = SquashedNormal(mu, std)
        return dist


class BigCritic(nn.Module):
    def __init__(self, state_space_size=15, action_space_size=8, hidden_size=1024):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(state_space_size+action_space_size, hidden_size),

                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),

                                 nn.ReLU())

        #self.fc3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),

                                 #nn.ReLU())
        self.fc3 = nn.Linear(hidden_size, 1)

        self.apply(weight_init)

    def forward(self, state, action):
        x = self.fc1(torch.cat((state, action), dim=1))
        x = self.fc2(x)
        #x = self.fc3(x)
        out = self.fc3(x)
        return out

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-0.99, 0.99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu



