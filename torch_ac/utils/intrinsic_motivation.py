#!/usr/bin/env python3

import math
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CountModule():
    def __init__(self, state_action = False):
        self.state_action = state_action # define if depends also to the state
        self.counts = {}

    def visualize_counts(self):
        print('\nDict count:')
        for e,(k,v) in enumerate(self.counts.items()):
            print('Hash state:{} of len {} with Values:{}'.format(e,len(k),v))

    def check_ifnot_already_visited(self,next_obs,actions):
        """
            Used to generate determine if a state has been visited
            (with inverse logic)
        """
        tup = self.encode_state(next_obs,actions)
        if tup in self.counts:
            return 0 # if visited, mask=0
        else:
            return 1 # if not visited, mask=1

    def compute_intrinsic_reward(self,obs,next_obs,coordinates,actions):
        """
            Generates the Intrinsic reward bonus based on the encoded state/tuple
            -Accepts a single observation
        """
        tup = self.encode_state(next_obs,actions)
        if tup in self.counts:
            return 1/math.sqrt(self.counts[tup])
        else:
            return 1

    def update(self,obs,next_obs,coordinates,actions):
        """
            Add samples to the bins;
                -It is prepared to catch inputs of shape [batch_size, -1]
                -i.e. [2048,7,7,3]

            Returns: Intrinsic Rewards Values
        """
        int_rew_tensor = np.ones(len(actions))

        for idx,(o,a) in enumerate(zip(next_obs,actions)):
            tup = self.encode_state(o,a)
            if tup in self.counts:
                self.counts[tup] += 1
                v = 1/math.sqrt(self.counts[tup])
            else:
                self.counts[tup] = 1
                v = 1

            int_rew_tensor[idx] = v
        return int_rew_tensor

    def encode_state(self,state,action):
        """
            Encodes the state in a tuple or taking also into account the action
        """
        state = state.view(-1).tolist()
        if self.state_action:
            return (tuple(state),action)
        else:
            return (tuple(state))

    def reset(self):
        """
            Re-init of counts
        """
        self.counts = {}

class RNDModule():
    def __init__(self, device, neural_network_predictor, neural_network_target, learning_rate = 0.0001, state_action = False):

        self.device = device
        self.state_action = state_action # define if depends also to the state
        self.update_proportion = 1.0

        # RND networks
        self.predictor = neural_network_predictor
        self.target = neural_network_target

        self.optimizer = optim.Adam(list(self.predictor.parameters()),
                                    lr=learning_rate)
        # move to GPU/CPU
        self.predictor = self.predictor.to(self.device)
        self.target = self.target.to(self.device)

        self.predictor.train()
        self.target.eval()

        self.forward_mse = nn.MSELoss(reduction='none')


    def preprocess_observations(self,input_obs):

        # check if it is numpy or tensor and convert to tensor
        if not torch.is_tensor(input_obs):
            input_obs = torch.tensor(input_obs,device=self.device,dtype=torch.float)

        # ensure 4 dims [batch, 7, 7, 3]
        if input_obs.dim() == 3: # only one observation, is not a batch
            input_obs = input_obs.unsqueeze(0)

        # reshape to be [batch, 3, 7, 7]
        input_obs = input_obs.transpose(1, 3).transpose(2, 3)

        return input_obs

    def compute_intrinsic_reward(self,obs,next_obs,coordinates,actions):
        """
            Genrate Intrinsic reward bonus based on the given input
        """
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(next_obs)

        with torch.no_grad():
            predict_next_state_feature = self.predictor(input_obs)
            target_next_state_feature = self.target(input_obs)

        # intrinsic_reward = torch.norm(predict_next_state_feature.detach() - target_next_state_feature.detach(), dim=1, p=2)
        intrinsic_reward = self.forward_mse(predict_next_state_feature,target_next_state_feature).mean(-1)

        # print('INt rw new:',intrinsic_reward)
        # print('INt rw old:',intrinsic_reward_old)
        # input('\n')

        return intrinsic_reward.item()

    def update(self,obs,next_obs,coordinates,actions):
        """
            Update NN parameters with batch of observations
        """
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(next_obs)

        predict_next_state_feature = self.predictor(input_obs)
        with torch.no_grad():
            target_next_state_feature = self.target(input_obs)

        # compute loss
        forward_loss = self.forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
        # forward_loss = torch.norm(predict_next_state_feature - target_next_state_feature, dim=1, p=2)#.mean(-1)
        # print('pred',predict_next_state_feature.shape)
        # print('target',target_next_state_feature.shape)
        # print('fw',forward_loss)
        # print('fw norm:',aux)


        # Proportion of exp used for predictor update (select randomly the samples collected by a groupd of parallel envs)
        mask = torch.rand(len(forward_loss)).to(self.device)
        mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
        # update loss to be proportional
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

        # Optimization step
        self.optimizer.zero_grad()
        forward_loss.backward()
        # grad_normalization
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 0.5)

        self.optimizer.step()

class ICMModule():
    def __init__(self, device, emb_network, inv_network, forw_network, learning_rate = 0.0001):

        self.device = device

        # pathak used Beta = 0.2 and landa=0.1
        self.forward_loss_coef = 10
        self.inverse_loss_coef = 0.1
        # networks
        self.state_embedding = emb_network
        self.inverse_dynamics = inv_network
        self.forward_dynamics = forw_network

        self.optimizer_state_embedding = optim.Adam(list(self.state_embedding.parameters()),
                                         lr=learning_rate)
        self.optimizer_inverse_dynamics = optim.Adam(list(self.inverse_dynamics.parameters()),
                                         lr=learning_rate)
        self.optimizer_forward_dynamics = optim.Adam(list(self.forward_dynamics.parameters()),
                                         lr=learning_rate)
        # move to GPU/CPU
        self.state_embedding = self.state_embedding.to(self.device)
        self.inverse_dynamics = self.inverse_dynamics.to(self.device)
        self.forward_dynamics = self.forward_dynamics.to(self.device)

        self.state_embedding.train()
        self.inverse_dynamics.train()
        self.forward_dynamics.train()

    def preprocess_observations(self,input_obs):

        # check if it is numpy or tensor and convert to tensor
        if not torch.is_tensor(input_obs):
            input_obs = torch.tensor(input_obs,device=self.device,dtype=torch.float)

        # ensure 4 dims [batch, 7, 7, 3]
        if input_obs.dim() == 3: # only one observation, is not a batch
            input_obs = input_obs.unsqueeze(0)

        # reshape to be [batch, 3, 7, 7]
        input_obs = input_obs.transpose(1, 3).transpose(2, 3)

        return input_obs

    def compute_intrinsic_reward(self,obs,next_obs,coordinates,actions):
        """
            Genrate Intrinsic reward bonus based on the given input
        """
        # print('Compute Intrinsic Reward At ICM Module')
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        with torch.no_grad():
            # s' after embedding network
            next_state_emb = self.state_embedding(input_next_obs)

            # state embedding of s for forward dynamic model
            state_emb = self.state_embedding(input_obs) # returns [batch,feat_output_dim=32]
            # prediction of s' taken into account also actual environment action
            act = actions.unsqueeze(0).unsqueeze(0) # add dimensions for the scalar to become a "list"; then add batch dim
            pred_next_state_emb = self.forward_dynamics(state_emb,act)
            # print('state emb {}; pred_next_state: {}'.format(state_emb.shape,pred_next_state_emb.shape))


        # Calculate intrinsic rewards; we get [batch, num_feature_prediction]
        intrinsic_reward = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2)
        # print('INt rw new:',intrinsic_reward)

        return intrinsic_reward.item()

    def update(self,obs,next_obs,coordinates,actions):
        """
            Update NN parameters with batch of observations
        """
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        # get s and s'
        state_emb = self.state_embedding(input_obs)
        next_state_emb = self.state_embedding(input_next_obs)

        # *********************************************************************
        # 1. loss of inverse module
        pred_actions = self.inverse_dynamics(state_emb, next_state_emb)
        true_actions = actions

        # pre-process
        log_soft_pred_actions = F.log_softmax(pred_actions,dim=1)
        true_actions = true_actions.long() # int tensor type; Long is required for nll_loss
        print('log_soft_pref_actions:',log_soft_pred_actions)
        # generate cross_entropy/nll_loss
        # expected input to be:
        #  - log_soft_pred_actions --> [batch,action_size] action_size are the log probs for each action
        #  - true_actions --> [batch]
        inverse_dynamics_loss = F.nll_loss(log_soft_pred_actions,
                                           target = true_actions.flatten(),
                                           reduction='none')
        # print('cross entropy loss:', inverse_dynamics_loss.shape)

        # finally get the avg loss of the whole batch
        inverse_dynamics_loss = torch.mean(inverse_dynamics_loss, dim=0)
        # *********************************************************************

        # *********************************************************************
        # 2. loss of forward module
        pred_next_state_emb = self.forward_dynamics(state_emb,actions)
        forward_dynamics_loss = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2)
        # print('pred next_state_embedding:',pred_next_state_emb.shape)
        # print('next_state_embedding:',next_state_emb.shape)
        # print('fwloss.shape:',forward_dynamics_loss.shape)
        forward_dynamics_loss = torch.mean(forward_dynamics_loss, dim=0)
        # *********************************************************************

        # total loss
        icm_loss = self.forward_loss_coef * forward_dynamics_loss + self.inverse_loss_coef*inverse_dynamics_loss

        # Optimization step
        self.optimizer_state_embedding.zero_grad()
        self.optimizer_inverse_dynamics.zero_grad()
        self.optimizer_forward_dynamics.zero_grad()

        # backpropagation of gradients
        icm_loss.backward()

        # grad_clipping
        torch.nn.utils.clip_grad_norm_(self.state_embedding.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), 0.5)

        self.optimizer_state_embedding.step()
        self.optimizer_inverse_dynamics.step()
        self.optimizer_forward_dynamics.step()

class RIDEModule():
    def __init__(self, device, emb_network, inv_network, forw_network, learning_rate = 0.0001):
        """
            Same implementation as ICM, but the compute_intrinic_reward slightly changes
        """

        self.device = device

        # pathak used Beta = 0.2 and landa=0.1
        self.forward_loss_coef = 10
        self.inverse_loss_coef = 0.1
        # networks
        self.state_embedding = emb_network
        self.inverse_dynamics = inv_network
        self.forward_dynamics = forw_network

        self.optimizer_state_embedding = optim.Adam(list(self.state_embedding.parameters()),
                                         lr=learning_rate)
        self.optimizer_inverse_dynamics = optim.Adam(list(self.inverse_dynamics.parameters()),
                                         lr=learning_rate)
        self.optimizer_forward_dynamics = optim.Adam(list(self.forward_dynamics.parameters()),
                                         lr=learning_rate)
        # move to GPU/CPU
        self.state_embedding = self.state_embedding.to(self.device)
        self.inverse_dynamics = self.inverse_dynamics.to(self.device)
        self.forward_dynamics = self.forward_dynamics.to(self.device)

        self.state_embedding.train()
        self.inverse_dynamics.train()
        self.forward_dynamics.train()

    def preprocess_observations(self,input_obs):

        # check if it is numpy or tensor and convert to tensor
        if not torch.is_tensor(input_obs):
            input_obs = torch.tensor(input_obs,device=self.device,dtype=torch.float)

        # ensure 4 dims [batch, 7, 7, 3]
        if input_obs.dim() == 3: # only one observation, is not a batch
            input_obs = input_obs.unsqueeze(0)

        # reshape to be [batch, 3, 7, 7]
        input_obs = input_obs.transpose(1, 3).transpose(2, 3)

        return input_obs

    def compute_intrinsic_reward(self,obs,next_obs,coordinates,actions):
        """
            Genrate Intrinsic reward bonus based on the given input
        """
        # print('Compute Intrinsic Reward At ICM Module')
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        with torch.no_grad():
            # state embedding of s for forward dynamic model
            state_emb = self.state_embedding(input_obs) # returns [batch,feat_output_dim=32]
            # s' after embedding network
            next_state_emb = self.state_embedding(input_next_obs)

        # Calculate intrinsic rewards; we get [batch, num_feature_prediction]
        intrinsic_reward = torch.norm(next_state_emb - state_emb, dim=1, p=2)
        # print('INt rw new:',intrinsic_reward)

        return intrinsic_reward.item()

    def update(self,obs,next_obs,coordinates,actions):
        """
            Update NN parameters with batch of observations
        """
        # get tensor shape [batch,3,7,7]
        input_obs = self.preprocess_observations(obs)
        input_next_obs = self.preprocess_observations(next_obs)

        # get s and s'
        state_emb = self.state_embedding(input_obs)
        next_state_emb = self.state_embedding(input_next_obs)

        # *********************************************************************
        # 1. loss of inverse module
        pred_actions = self.inverse_dynamics(state_emb, next_state_emb)
        true_actions = actions

        # print('pred_actions', pred_actions.shape)
        # print('true_actions', true_actions.shape)

        # pre-process
        log_soft_pred_actions = F.log_softmax(pred_actions,dim=1)
        true_actions = true_actions.long() # int tensor type; Long is required for nll_loss

        # generate cross_entropy/nll_loss
        # expected input to be:
        #  - log_soft_pred_actions --> [batch,action_size] action_size are the log probs for each action
        #  - true_actions --> [batch]
        inverse_dynamics_loss = F.nll_loss(log_soft_pred_actions,
                                           target = true_actions.flatten(),
                                           reduction='none')
        # print('cross entropy loss:', inverse_dynamics_loss.shape)

        # finally get the avg loss of the whole batch
        inverse_dynamics_loss = torch.mean(inverse_dynamics_loss, dim=0)
        # *********************************************************************

        # *********************************************************************
        # 2. loss of forward module
        pred_next_state_emb = self.forward_dynamics(state_emb,actions)
        forward_dynamics_loss = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=2)
        # print('pred next_state_embedding:',pred_next_state_emb.shape)
        # print('next_state_embedding:',next_state_emb.shape)
        # print('fwloss.shape:',forward_dynamics_loss.shape)
        forward_dynamics_loss = torch.mean(forward_dynamics_loss, dim=0)
        # *********************************************************************

        # total loss
        icm_loss = self.forward_loss_coef * forward_dynamics_loss + self.inverse_loss_coef*inverse_dynamics_loss

        # Optimization step
        self.optimizer_state_embedding.zero_grad()
        self.optimizer_inverse_dynamics.zero_grad()
        self.optimizer_forward_dynamics.zero_grad()

        # backpropagation of gradients
        icm_loss.backward()

        # grad_clipping
        torch.nn.utils.clip_grad_norm_(self.state_embedding.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.inverse_dynamics.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.forward_dynamics.parameters(), 0.5)

        self.optimizer_state_embedding.step()
        self.optimizer_inverse_dynamics.step()
        self.optimizer_forward_dynamics.step()
