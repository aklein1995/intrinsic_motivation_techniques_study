import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
# def init_params(m):
#     classname = m.__class__.__name__
#     if classname.find("Linear") != -1:
#         m.weight.data.normal_(0, 1)
#         m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
#         if m.bias is not None:
#             m.bias.data.fill_(0)

# RIDE initilization
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ActorModel_RAPID(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        input_size=7*7*3
        # init
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                    constant_(x, 0), nn.init.calculate_gain('relu'))

        self.fc = nn.Sequential(
            init_(nn.Linear(input_size,64)),
            nn.ReLU(),
            init_(nn.Linear(64,64)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        # Define actor's model
        self.actor = nn.Sequential(
            init_(nn.Linear(64, action_space))
        )

    def forward(self, obs, memory=[]):
        obs_flatened = obs.image.view(obs.image.shape[0], -1)
        x = self.fc(obs_flatened)
        x = self.actor(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if len(memory)>0:
            return dist, memory
        else:
            return dist

class CriticModel_RAPID(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()

        input_size=7*7*3

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.fc = nn.Sequential(
            init_(nn.Linear(input_size,64)),
            nn.ReLU(),
            init_(nn.Linear(64,64)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        # Define critic's model
        self.critic_ext = nn.Sequential(
            init_(nn.Linear(64, 1))
        )
        self.critic_int = nn.Sequential(
            init_(nn.Linear(64, 1))
        )

    def forward(self, obs, memory=[]):
        obs_flatened = obs.image.view(obs.image.shape[0], -1)
        x = self.fc(obs_flatened)
        embedding = self.fc(obs_flatened)

        x = self.critic_ext(embedding)
        value_ext = x.squeeze(1)

        x = self.critic_int(embedding)
        value_int = x.squeeze(1)

        if len(memory) > 0:
            return value_ext, memory
        else:
            return value_ext

        # if len(memory)>0:
        #     return value_ext, value_int, memory
        # else:
        #     return value_ext, value_int

class ACModelRIDE(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_intcoefs=0, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        self.use_intcoefs = use_intcoefs #default = False

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define image embedding
        self.image_conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
        )
        n,m = obs_space["image"][0],obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_embedding_size = 32

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # add more deep (instead of LSTM)
        self.fc = nn.Sequential(
            init_(nn.Linear(self.embedding_size+self.use_intcoefs,256)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        # Define actor's model
        self.actor = nn.Sequential(
            init_(nn.Linear(256, action_space)),
        )

        # Define critic's model
        self.critic_ext = nn.Sequential(
            init_(nn.Linear(256, 1))
        )
        self.critic_int = nn.Sequential(
            init_(nn.Linear(256, 1))
        )

        # Initialize parameters correctly
        # self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, int_coefs=[], memory=[]):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        elif self.use_intcoefs:
            concat = torch.cat((x,int_coefs.unsqueeze(1)),dim=1)
            embedding = self.fc(concat)
        else:
            embedding = self.fc(x)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic_ext(embedding)
        value_ext = x.squeeze(1)

        # x = self.critic_int(embedding)
        # value_int = x.squeeze(1)


        if len(memory)>0:
            return dist, value_ext, memory
        else:
            return dist, value_ext
        # if len(memory)>0:
        #     return dist, value_ext, value_int, memory
        # else:
        #     return dist, value_ext, value_int
