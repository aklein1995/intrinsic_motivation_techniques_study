import torch

import utils
from .other import device
from model import *


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False,
                 separated_networks=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.separated_networks = separated_networks
        self.use_recurrence = use_memory
        self.action_space = action_space

        print('Model directory:',model_dir)
        print('self.separated_networks',self.separated_networks)
        print('Action space:',action_space)

        if self.separated_networks:
            actor = ActorModel_RAPID(obs_space, action_space, use_memory=use_memory)
            critic = CriticModel_RAPID(obs_space, action_space, use_memory=use_memory)

            actor.load_state_dict(utils.get_model_state(model_dir=model_dir,separated_network='actor'))
            critic.load_state_dict(utils.get_model_state(model_dir=model_dir,separated_network='critic'))

            actor.to(device)
            critic.to(device)

            actor.eval()
            critic.eval()

            self.acmodel = (actor, critic)
        else:
            self.acmodel = ACModelRIDE(obs_space, action_space, use_memory=use_memory, use_text=use_text)
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
            self.acmodel.to(device)
            self.acmodel.eval()

        self.argmax = argmax
        self.num_envs = num_envs

        if self.use_recurrence:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)


        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.use_recurrence:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                if self.separated_networks==True:
                    dist = self.acmodel[0](preprocessed_obss)
                else:
                    dist, _, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.use_recurrence:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
