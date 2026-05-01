import torch
from nets import StochasticActor
from nets import BigCritic


actor=StochasticActor(state_space_size=15,action_space_size=8)
torch.save(actor.state_dict(),'actor.pt')
critic1=BigCritic(state_space_size=15,action_space_size=8)
torch.save(critic1.state_dict(),'critic1.pt')
critic2=BigCritic(state_space_size=15,action_space_size=8)
torch.save(critic1.state_dict(),'critic2.pt')

torch.save(torch.tensor([0.1]),'alpha.pt')

