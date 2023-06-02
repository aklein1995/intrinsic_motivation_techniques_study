import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# RIDE initilization
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class EmbeddingNetwork_RIDE(nn.Module):
    """
     Based on the architectures selected at minigrid in RIDE:
     https://github.com/facebookresearch/impact-driven-exploration/blob/877c4ea530cc0ca3902211dba4e922bf8c3ce276/src/models.py#L352    """
    def __init__(self):
        super().__init__()

        input_size=7*7*3
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feature_extractor = nn.Sequential(
            init_(nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1)),
            nn.ELU(),
        )
        # params = sum(p.numel() for p in self.modules.parameters())
        # print('Params:',params)



    def forward(self, next_obs):
        feature = self.feature_extractor(next_obs)
        reshape = feature.view(feature.size(0),-1)

        return reshape

class EmbeddingNetwork_RAPID(nn.Module):
    """
     Based on the architectures selected at minigrid in rapid, 64-64 MLP:
    """
    def __init__(self):
        super().__init__()
        input_size=7*7*3
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feature_extractor = nn.Sequential(
            init_(nn.Linear(input_size,64)),
            nn.ReLU(),
            init_(nn.Linear(64,64)),
            nn.ReLU()
        )

    def forward(self, next_obs):
        obs_flatened = next_obs.reshape(next_obs.shape[0], -1) #https://discuss.pytorch.org/t/difference-between-view-reshape-and-permute/54157/2
        feature = self.feature_extractor(obs_flatened)

        return feature

class InverseDynamicsNetwork_RIDE(nn.Module):
    def __init__(self, num_actions, input_size = 32, device = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.input_size, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))


    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=1)

        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class InverseDynamicsNetwork_RAPID(nn.Module):
    def __init__(self, num_actions, input_size = 64, device = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device

        init_ = lambda m: init(m, nn.init.orthogonal_,
                lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * self.input_size, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 64)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(64, self.num_actions))


    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=1)
        # print('s shape:',state_embedding.shape)
        # print('ss shape:',next_state_embedding.shape)
        # print('inputs shape:',inputs.shape)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class ForwardDynamicsNetwork_RIDE(nn.Module):
    def __init__(self, num_actions, input_size=32, device = 'cpu'):
        """
            input_size depends on the output of the embedding network
        """
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.input_size + self.num_actions, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        # the loss of ICM is between two latent predictions; therefore, they have to have the same dimensiones
        # as the Embedding Network output is used here as input, we use it to determine this NN output
        self.fd_out = init_(nn.Linear(256, self.input_size))

    def forward(self, state_embedding, action):
        """
            INPUTS:
            -Action: it can be a single item value (when computing rewards) or
                    batch of values when updating [batch,action_taken]
            -State-embedding:
        """
        actions_one_hot = torch.zeros((action.shape[0],self.num_actions), device=self.device)

        # generate one-hot encoding of action
        for i,a in enumerate(action):
            a = a.squeeze(0) # we need a scalar tensor value, not a list for the transformation of one-hot
            a = a.long()# the transformation requires to be Long type
            one_hot = F.one_hot(a, num_classes=self.num_actions).float()
            actions_one_hot[i].copy_(one_hot)

        # concat and generate the input
        inputs = torch.cat((state_embedding, actions_one_hot),dim=1)
        # forward pass
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))

        return next_state_emb

class ForwardDynamicsNetwork_RAPID(nn.Module):
    def __init__(self, num_actions, input_size=64, device = 'cpu'):
        """
            input_size depends on the output of the embedding network
        """
        super().__init__()
        self.input_size = input_size
        self.num_actions = num_actions
        self.device = device


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.input_size + self.num_actions, 64)),
            nn.ReLU(),
            init_(nn.Linear(64, 64)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        # the loss of ICM is between two latent predictions; therefore, they have to have the same dimensiones
        # as the Embedding Network output is used here as input, we use it to determine this NN output
        self.fd_out = init_(nn.Linear(64, self.input_size))

    def forward(self, state_embedding, action):
        """
            INPUTS:
            -Action: it can be a single item value (when computing rewards) or
                    batch of values when updating [batch,action_taken]
            -State-embedding:
        """
        actions_one_hot = torch.zeros((action.shape[0],self.num_actions), device=self.device)

        # generate one-hot encoding of action
        for i,a in enumerate(action):
            a = a.squeeze(0) # we need a scalar tensor value, not a list for the transformation of one-hot
            a = a.long()# the transformation requires to be Long type
            one_hot = F.one_hot(a, num_classes=self.num_actions).float()
            actions_one_hot[i].copy_(one_hot)

        # concat and generate the input
        inputs = torch.cat((state_embedding, actions_one_hot),dim=1)
        # forward pass
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))

        return next_state_emb
