import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.0001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, num_actions = 7,
                 separated_networks=False, env_name='MiniGrid-DoorKey-5x5-v0',
                 int_coef=0.0, normalize_int_rewards = 0,
                 im_type = 'counts', int_coef_type='static',
                 use_episodic_counts=0, use_only_not_visited=0,total_num_frames=20000000,
                 reduced_im_networks=0):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                         separated_networks,env_name, num_actions, int_coef, normalize_int_rewards,
                         im_type,int_coef_type,use_episodic_counts,use_only_not_visited,total_num_frames,reduced_im_networks)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.int_coef = int_coef
        self.int_coef_type = int_coef_type

        assert self.batch_size % self.recurrence == 0

        # self.separated_actor_critic = separated_networks
        if self.separated_actor_critic:
            self.optimizer = [torch.optim.Adam(self.acmodel[i].parameters(), lr, eps=adam_eps)
                                for i in range(len(self.acmodel))]
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

        self.not_use_value_clipping = True #compute critic loss with MSE
        self.forward_mse = nn.MSELoss()


    def update_parameters(self, exps):
        # Collect experiences
        for _ in range(self.epochs):
            # Initialize log values
            log_entropies = []
            log_values = []
            #gradients
            log_grad_norms = []
            log_grad_norms_critic = []
            # losses
            log_policy_losses = []
            log_value_losses = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values
                # policy
                batch_entropy = 0
                batch_policy_loss = 0
                # critic
                batch_value = 0
                batch_value_loss = 0
                # both
                batch_loss = 0

                # Initialize memory

                if self.use_recurrence:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.use_recurrence:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        if self.separated_actor_critic:
                            dist = self.acmodel[0](sb.obs) #actor
                            value_ext = self.acmodel[1](sb.obs) #critic
                        else:
                            if self.int_coef_type == 'ngu':
                                dist, value_ext = self.acmodel(sb.obs,sb.int_coefs)
                            else:
                                dist, value_ext = self.acmodel(sb.obs)

                    # Policy related
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage_total
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage_total
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Critic related
                    if self.not_use_value_clipping:
                        value_loss = self.forward_mse(value_ext, sb.returnn_ext)
                    else:
                        # only extrinsic head
                        value_clipped = sb.value_ext + torch.clamp(value_ext - sb.value_ext, -self.clip_eps, self.clip_eps)
                        surr1 = self.forward_mse(value_ext,sb.returnn_ext)
                        surr2 = self.forward_mse(value_clipped,sb.returnn_ext)
                        value_loss = torch.max(surr1, surr2).mean()

                    # total loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values
                    #policy
                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    #critic
                    batch_value += value_ext.mean().item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.use_recurrence and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_policy_loss /= self.recurrence

                batch_value /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                if self.separated_actor_critic:

                    self.optimizer[0].zero_grad()
                    self.optimizer[1].zero_grad()
                    batch_loss.backward()
                    # grad_norm_actor_before = self.calculate_gradients(self.acmodel[0])
                    # grad_norm_critic_before = self.calculate_gradients(self.acmodel[1])
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.acmodel[0].parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.acmodel[1].parameters(), self.max_grad_norm)
                    grad_norm = grad_norm_actor = self.calculate_gradients(self.acmodel[0])
                    grad_norm_critic = self.calculate_gradients(self.acmodel[1])
                    self.optimizer[0].step()
                    self.optimizer[1].step()

                    # print('grads actor before:',grad_norm_actor_before)
                    # print('grads actor after:',grad_norm_actor)
                    #
                    # print('grads critic before:',grad_norm_critic_before)
                    # print('grads critic after:',grad_norm_critic)
                    # input()
                else:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    # grad_norm_before = self.calculate_gradients(self.acmodel)

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                    grad_norm = self.calculate_gradients(self.acmodel)
                    grad_norm_critic = 0
                    self.optimizer.step()

                    # print('grads before:',grad_norm_before)
                    # print('grads after:',grad_norm)
                    # input()
                # Update log values

                log_entropies.append(batch_entropy)
                log_policy_losses.append(batch_policy_loss)

                log_values.append(batch_value)
                log_value_losses.append(batch_value_loss)

                log_grad_norms.append(grad_norm) #used to monitore either AC or just the actor
                log_grad_norms_critic.append(grad_norm_critic) # when having 2 networks, we monitore the critic here

        # Log some values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "policy_loss": numpy.mean(log_policy_losses),
            "value": numpy.mean(log_values),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "grad_norm_critic": numpy.mean(log_grad_norms_critic)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def calculate_gradients(self, model):
        """
        Given the current network with its graph, it calculates the gradients through
        each module and returns a sum of values

        By default, the network will have two-head of critic but we will propagate only
        through one or the both
        """
        # if self.two_value_heads and self.int_coef>0:
        if False:
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5 #monitorization of gradients
        else:
            # when using two value heads but we do not propagate through one of them, there are some gradients=None and arise errors
            grad_norm = 0
            for enum,p in enumerate(model.parameters()):
                # print('{} Parameter shape: {}'.format(enum,p.shape))
                # print('Grad:',p.grad)
                try:
                    grad_norm += p.grad.data.norm(2).item() ** 2
                except AttributeError:
                    # print('No grad')
                    continue
            grad_norm = grad_norm **0.5

        return grad_norm
