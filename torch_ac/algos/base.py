from abc import ABC, abstractmethod
import time
import torch
import numpy as np
from copy import deepcopy
from collections import deque
from statistics import mean

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch_ac.utils import RunningMeanStd,RewardForwardFilter,yvalue_richard_curve
from torch_ac.utils.intrinsic_motivation import CountModule,RNDModule,ICMModule,RIDEModule
from torch_ac.utils.im_models import EmbeddingNetwork_RAPID, EmbeddingNetwork_RIDE, \
                                        InverseDynamicsNetwork_RAPID, InverseDynamicsNetwork_RIDE, \
                                        ForwardDynamicsNetwork_RAPID, ForwardDynamicsNetwork_RIDE

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward,
                 separated_networks, env_name, num_actions,
                 int_coef, normalize_int_rewards,
                 im_type, int_coef_type,use_episodic_counts,use_only_not_visited,
                 total_num_frames,reduced_im_networks):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module or tuple of torch.Module(s)
            the model(s); the separated_actor_critic parameter defines that
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        separated_networks: boolean
            set whether we are going to use a single AC neural network or
            two differents
        """

        # Store parameters

        self.separated_actor_critic = separated_networks
        self.reduced_im_networks = reduced_im_networks

        self.use_recurrence = True if recurrence > 1 else False
        self.acmodel = acmodel

        self.num_actions = num_actions
        self.env = ParallelEnv(envs,env_name)
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Control parameters

        assert self.use_recurrence or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        if self.separated_actor_critic:
            for i in range(len(self.acmodel)):
                self.acmodel[i].to(self.device)
                self.acmodel[i].train()
        else:
            self.acmodel.to(self.device)
            self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.total_num_frames = total_num_frames
        print('total frames:',total_num_frames)
        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.agent_position = [None]*(shape[0])

        if self.use_recurrence:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values_ext = torch.zeros(*shape, device=self.device)
        self.advantages_ext = torch.zeros(*shape, device=self.device)
        self.returns_ext = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize LOGs values
        self.log_episode_return = torch.zeros(self.num_procs, device=self.device) # monitores the return inside the episode (it increases with each step until done is reached)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs # monitores the total return that was given in the whole episode (updates after each episode)
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        self.episode_counter = 0
        self.frames_counter = 0

        # for intrinsic coef adaptive decay
        self.log_rollout_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_train = torch.tensor([],device=self.device) # stores the avg return reported after each rollout (avg of all penv after every nsteps)

        # *****  Intrinsic motivation related parameters *****
        self.int_coef = int_coef
        self.use_normalization_intrinsic_rewards = normalize_int_rewards


        self.im_type = im_type # module type ['rnd', 'icm', 'counts']
        self.int_coef_type = int_coef_type # ['static','parametric','adaptive']
        # define IM module
        if self.im_type == 'counts':
            print('\nUsing COUNTS')
            self.im_module = CountModule()

        elif self.im_type == 'rnd':
            print('\nUsing RND')
            if self.reduced_im_networks:
                self.im_module = RNDModule(neural_network_predictor=EmbeddingNetwork_RAPID(),
                                      neural_network_target=EmbeddingNetwork_RAPID(),
                                      device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RAPID().parameters())
                total_params = embedding_params*2
                print('***PARAMS with RAPID ARCHITECTURE:\nEmbedding {}\nTotal {}\n'.format(embedding_params,total_params))
            else:
                self.im_module = RNDModule(neural_network_predictor=EmbeddingNetwork_RIDE(),
                                      neural_network_target=EmbeddingNetwork_RIDE(),
                                      device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RIDE().parameters())
                total_params = embedding_params*2
                print('***PARAMS with RIDE ARCHITECTURE:\nEmbedding {}\nTotal {}\n'.format(embedding_params,total_params))
        elif self.im_type == 'icm':
            print('\nUsing ICM')
            if self.reduced_im_networks:
                self.im_module = ICMModule(emb_network = EmbeddingNetwork_RAPID(),
                                           inv_network = InverseDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           device = device)
            else:
                self.im_module = ICMModule(emb_network = EmbeddingNetwork_RIDE(),
                                           inv_network = InverseDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           device = device)

        elif self.im_type == 'ride':
            print('\nUsing RIDE')
            if self.reduced_im_networks:
                self.im_module = RIDEModule(emb_network = EmbeddingNetwork_RAPID(),
                                           inv_network = InverseDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions, device=self.device),
                                           device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RAPID().parameters())
                inv_dynamics_params = sum(p.numel() for p in InverseDynamicsNetwork_RAPID(num_actions=self.num_actions).parameters())
                forw_dynamics_params = sum(p.numel() for p in ForwardDynamicsNetwork_RAPID(num_actions=self.num_actions).parameters())
                total_params = embedding_params + inv_dynamics_params + forw_dynamics_params
                print('***PARAMS with RAPID ARCHITECTURE:\nEmbedding {}\nInverse {}\nForward {}\nTotal {}\n'.format(embedding_params,inv_dynamics_params,forw_dynamics_params,total_params))
            else:
                self.im_module = RIDEModule(emb_network = EmbeddingNetwork_RIDE(),
                                           inv_network = InverseDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           forw_network = ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions, device=self.device),
                                           device = device)
                embedding_params = sum(p.numel() for p in EmbeddingNetwork_RIDE().parameters())
                inv_dynamics_params = sum(p.numel() for p in InverseDynamicsNetwork_RIDE(num_actions=self.num_actions).parameters())
                forw_dynamics_params = sum(p.numel() for p in ForwardDynamicsNetwork_RIDE(num_actions=self.num_actions).parameters())
                total_params = embedding_params + inv_dynamics_params + forw_dynamics_params
                print('***PARAMS with RIDE ARCHITECTURE:\nEmbedding {}\nInverse {}\nForward {}\nTotal {}\n'.format(embedding_params,inv_dynamics_params,forw_dynamics_params,total_params))

        # episodic counts and first visit variables
        self.use_episodic_counts = 1 if im_type == 'ride' else use_episodic_counts # ride always uses episodic counts by default
        self.episodic_counts = [CountModule() for _ in range(self.num_procs)] # counts used to carry out how many times each observation has been visited inside an episode
        self.use_only_not_visited = use_only_not_visited
        self.visited_state_in_episode = torch.zeros(*shape, device=self.device) # mask that is used to allow or not compute a non-zero intrinsic reward

        # Parameters needed when using two-value/advantage combination for normalization
        self.return_rms = RunningMeanStd()
        self.normalization_int_score = 0
        self.min_std = 0.01
        self.predominance_ext_over_int = torch.zeros(*shape, device=self.device)

        # experience values
        self.rewards_int = torch.zeros(*shape, device=self.device)
        self.rewards_total = torch.zeros(*shape, device=self.device)
        self.advantages_int = torch.zeros(*shape, device=self.device)
        self.advantages_total = torch.zeros(*shape, device=self.device)
        self.returns_int = torch.zeros(*shape, device=self.device)
        # add monitorization for intrinsic part
        self.log_episode_return_int = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int =  [0] * self.num_procs
        # other for normalization
        self.log_episode_return_int_normalized = torch.zeros(self.num_procs, device=self.device)
        self.log_return_int_normalized =  [0] * self.num_procs
        # add avg 100 episodes return
        self.last_100return = deque([0],maxlen=100)
        self.ngu_episode_return = 0

        print('num_frame per proc:',self.num_frames_per_proc)
        print('num of process:',self.num_procs)
        print('num frames (num_pallel envs*framesperproc):', self.num_frames)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        forw_time_total = []

        for i in range(self.num_frames_per_proc):

            # update frame counter after each step
            self.frames_counter += self.num_procs

            if self.int_coef_type == 'ngu':
                beta_values = [yvalue_richard_curve(im_coef=self.int_coef,im_type=self.im_type,max_steps=self.num_procs,timestep=i) for i in range(self.num_procs)]
                actual_int_coef = torch.tensor(beta_values,device=self.device).float()
                actual_int_coef[-1] = 0
                # print('int coef',actual_int_coef)
                # input()
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.use_recurrence:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    if self.separated_actor_critic:
                        dist = self.acmodel[0](preprocessed_obs)
                        value = self.acmodel[1](preprocessed_obs)
                    else:
                        if self.int_coef_type == 'ngu':
                            dist, value = self.acmodel(obs=preprocessed_obs,int_coefs=actual_int_coef)
                        else:
                            dist, value = self.acmodel(obs=preprocessed_obs)

            # take action from distribution
            action = dist.sample()
            # execute step into environment
            obs, reward, done, agent_pos = self.env.step(action.cpu().numpy())

            # Update experiences values

            self.obss[i] = self.obs # stores the current observation on the experience
            self.obs = obs # stores the next_obs obtained after the step in the env
            self.agent_position[i] = agent_pos

            if self.use_recurrence:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values_ext[i] = value

            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # ***Intrinsic motivation - calculate intrinsic rewards
            if self.int_coef > 0:

                # calculate bonus (step by step)  - shape [num_procs, 7,7,3]
                input_current_obs = preprocessed_obs
                input_next_obs = self.preprocess_obss(self.obs, device=self.device) # contains next_observations

                # FOR COMPUTING INTRINSIC REWARD, THE REQUIRED SHAPE IS JUST A UNIT -- i.e image of [7,7,3]; action of [1] (it is calculated one by one)
                # FOR UPDATING COUNTS (done IN BATCH for efficiency), the shape requires to have the batch-- i.e image of [batch,7,7,3]; action of [batch,1]

                forw_time_i = time.time()
                rewards_int = [self.im_module.compute_intrinsic_reward(obs=ob,next_obs=nobs,coordinates=coords,actions=act) \
                                            for ob,nobs,coords,act in zip(input_current_obs.image, input_next_obs.image, agent_pos, action)]
                forw_time_f = time.time()
                forw_time_total.append(forw_time_f - forw_time_i)

                self.rewards_int[i] = rewards_int_torch = torch.tensor(rewards_int,device=self.device,dtype=torch.float)


                # Reward only when agent visits state s for the first time in the episode
                if self.use_only_not_visited:

                    # check if state has already been visited -- mask
                    self.visited_state_in_episode[i] = torch.tensor([self.episodic_counts[penv].check_ifnot_already_visited(next_obs=nobs,actions=act) \
                                                        for penv,(ob,nobs,coords,act) in enumerate(zip(input_current_obs.image, input_next_obs.image, agent_pos, action))])

                    self.rewards_int[i] = rewards_int_torch * self.visited_state_in_episode[i]

                # To use episodic counts (mandatory in the case of ride)
                if self.use_episodic_counts or self.use_only_not_visited:
                    # ***UPDATE EPISODIC COUNTER (mandatory for both episodic and 1st visitation count strategies)***
                    current_episode_count_reward = np.array([self.episodic_counts[penv].update(obs=ob,next_obs=nobs,coordinates=coords,actions=act) \
                                                    for penv,(ob,nobs,coords,act) in enumerate(zip(input_current_obs.image.unsqueeze(1), input_next_obs.image.unsqueeze(1) , agent_pos, action.unsqueeze(1).unsqueeze(1) ) )]) # we need to squeeze to have actions of shape [num_procs, 1, 1] and also the observations [num_procs,1,7,7,3]
                    # the update function returns the sqrt inverse value of counts (the intrinsic reward value in counts for the next_state)
                    if self.use_episodic_counts:
                    # Divide/multiply by episodic counts

                        current_episode_count_reward = torch.from_numpy(current_episode_count_reward).to(self.device)
                        self.rewards_int[i] = rewards_int_torch * current_episode_count_reward.squeeze(1)
                        # print('Actual rewards:',rewards_int_torch)
                        # print('Current episode counter:',current_episode_count_reward)
                        # print('Final Rewards:', self.rewards_int[i])

            # Update log values
            if self.int_coef > 0:
                self.log_episode_return_int += rewards_int_torch
                self.log_episode_return_int_normalized += torch.tensor(np.asarray(rewards_int)/max(self.normalization_int_score,self.min_std), device=self.device, dtype=torch.float)
                # used for adaptive int coef
                self.log_rollout_return_int += rewards_int_torch
                # print('log episode_return:',self.log_episode_return_int)
                # print('log rollout return',self.log_rollout_return_int)

            if self.int_coef_type == 'ngu':
                # we just take into account the last agent, which has beta=0
                self.ngu_episode_return += torch.tensor(reward[-1], device=self.device, dtype=torch.float)
                ngu_mask = self.mask[-1]

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)


            # for all the num_procs...
            for i, done_ in enumerate(done):
                if done_:
                    self.episodic_counts[i].reset()

                    # log related
                    if self.int_coef > 0:
                        self.log_return_int.append(self.log_episode_return_int[i].item())
                        self.log_return_int_normalized.append(self.log_episode_return_int_normalized[i].item())
                    self.log_done_counter += 1
                    self.episode_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    # save avg return of last 100 episodes
                    if self.int_coef_type == 'ngu' and (i == len(done) - 1):
                        self.last_100return.append(self.ngu_episode_return.item())
                        # if self.ngu_episode_return > 0:
                        # print('ngu score:',self.ngu_episode_return.item())
                    else:
                        self.last_100return.append(self.log_episode_return[i].item())

            # Intrinsic Return related
            if self.int_coef > 0:
                # resetea log a 0 si el episodio habia terminado (mask != done)
                self.log_episode_return_int *= self.mask
                self.log_episode_return_int_normalized *= self.mask

            if self.int_coef_type == 'ngu':
                self.ngu_episode_return *= ngu_mask

            self.log_episode_return *= self.mask #
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            # **********************************************************************
            # ONE STEP INSIDE THE ROLLOUT COMPLETED
            # **********************************************************************


        # **********************************************************************
        # ROLLOUT COLLECTION FINISHED.
        # **********************************************************************

        # 1.Update IM Module
        # 2.Normalize intrinsic rewards (before training)

        # Part 1 of updating...
        if self.int_coef > 0:
            # 1.1. preprocess the batch of data to be Tensors
            shape_im = (self.num_frames_per_proc,self.num_procs, 7,7,3) # preprocess batch observations (num_steps*num_instances, 7 x 7 x 3)
            input_obss = torch.zeros(*shape_im,device=self.device)
            input_nobss = torch.zeros(*shape_im,device=self.device)

            # generate next_states (same as self.obss + an additional next_state of al the penvs)
            nobss = deepcopy(self.obss)
            nobss = nobss[1:] # pop first element and move left
            nobss.append(self.obs) # add at the last position the next_states

            for num_frame,(mult_obs,mult_nobs) in enumerate(zip(self.obss,nobss)): # len(self.obss) ==> num_frames_per_proc == number_of_step

                for num_process,(obss,nobss) in enumerate(zip(mult_obs,mult_nobs)):
                    o = torch.tensor(obss['image'], device=self.device)
                    no = torch.tensor(nobss['image'], device=self.device)
                    input_obss[num_frame,num_process].copy_(o)
                    input_nobss[num_frame,num_process].copy_(no)

            # 1.2. reshape to have [num_frames*num_procs, 7, 7, 3]
            input_obss = input_obss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_nobss = input_nobss.view(self.num_frames_per_proc*self.num_procs,7,7,3)
            input_actions = self.actions.view(self.num_frames_per_proc*self.num_procs,-1)

            # self.im_module.visualize_counts()

            # 1.3. Update
            backw_time_i = time.time()
            self.im_module.update(obs=input_obss,next_obs=input_nobss,actions=input_actions,coordinates=self.agent_position)
            # Calculate times related to neural networks
            backw_time_f = time.time()
            backw_time_total = backw_time_f - backw_time_i
            # update forward time also
            forw_time_total = np.sum(forw_time_total)
            total_time_fwandbw = forw_time_total + backw_time_total

            # 2. Normalize (if required)
            if self.use_normalization_intrinsic_rewards:
                # Calculate normalization after each rollout
                batch_mean, batch_var, batch_count = self.log_rollout_return_int.mean(-1).item(), self.log_rollout_return_int.var(-1).item(), len(self.log_rollout_return_int)
                self.return_rms.update_from_moments(batch_mean, batch_var, batch_count)
                self.normalization_int_score = np.sqrt(self.return_rms.var)
                self.normalization_int_score = max(self.normalization_int_score, self.min_std)

                # apply normalization
                self.rewards_int /= self.normalization_int_score


        # obtain next_value for computing advantages and return
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        with torch.no_grad():
            if self.use_recurrence:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                if self.separated_actor_critic:
                    next_value = self.acmodel[1](preprocessed_obs)
                else:
                    if self.int_coef_type == 'ngu':
                        _, next_value = self.acmodel(obs=preprocessed_obs,int_coefs=actual_int_coef)
                    else:
                        _, next_value = self.acmodel(obs=preprocessed_obs)

        # **********************************************************************
        # Calculate new int coef(beta_t) based on static, parametric or adaptive decays
        # **********************************************************************
        hist_ret_avg = 1 #default value (to monitore)

        if self.int_coef_type == 'static':
            actual_int_coef = self.int_coef
        elif self.int_coef_type == 'parametric':
            actual_int_coef = yvalue_richard_curve(im_coef=self.int_coef,im_type=self.im_type,max_steps=self.total_num_frames,timestep=self.frames_counter)
        elif self.int_coef_type == 'adaptive' or self.int_coef_type == 'adaptive_1000':

            # update historic avg with the avg return collected by all the agents
            avg = self.log_rollout_return_int.mean(-1)
            self.log_return_int_train = torch.cat((self.log_return_int_train,avg.unsqueeze(0)), dim=0)

            # get historical return avg (same for all agents)
            if self.int_coef_type == 'adaptive_1000':
                hist_ret_avg = torch.mean(self.log_return_int_train[-1000:])
            else:
                hist_ret_avg = torch.mean(self.log_return_int_train)

            # one different actual_inf_coef for each agent
            decay_aux = torch.zeros(self.num_procs)
            for penv in range(self.num_procs):
                decay_aux[penv] = min(1, self.log_rollout_return_int[penv]/hist_ret_avg)

            actual_int_coef = self.int_coef*decay_aux
            hist_ret_avg = hist_ret_avg.item()

        # *** reinit as only has rollout scope
        self.log_rollout_return_int = torch.zeros(self.num_procs, device=self.device) # do it here, not anywhere else; adaptive method requires reset after calculation


        # ***Combining EXTRINSIC-INTRINSIC rewards***
        if self.int_coef <= 0:
            # No intrinsic_rewards
            self.rewards_total.copy_(self.rewards)

            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values_ext[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages_ext[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values_ext[i]
                self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            self.returns_ext =  self.values_ext + self.advantages_ext
            self.advantages_total.copy_(self.advantages_ext)

        # USING INTRINSIC MOTIVATION
        else:
            # 1. *** r_total = r_ext + Beta*r_int ***
            if self.int_coef_type == 'adaptive' or self.int_coef_type == 'adaptive_1000' or self.int_coef_type == 'ngu':
                for penv in range(self.num_procs):
                    self.rewards_total[:,penv] = self.rewards[:,penv] + actual_int_coef[penv]*self.rewards_int[:,penv]
                    self.rewards_total[:,penv] /= (1+actual_int_coef[penv])
            else:
                # more efficient (if non-adaptive used)
                self.rewards_total = self.rewards + actual_int_coef*self.rewards_int
                self.rewards_total /= (1+actual_int_coef)

            # 2. Calculate advantages and returns
            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values_ext[i+1] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages_ext[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards_total[i] + self.discount * next_value * next_mask - self.values_ext[i]
                self.advantages_ext[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

            self.returns_ext =  self.values_ext + self.advantages_ext
            self.advantages_total.copy_(self.advantages_ext)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.use_recurrence:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # extrinsic stream (used normally)
        exps.value_ext = self.values_ext.transpose(0, 1).reshape(-1)
        exps.advantage_ext = self.advantages_ext.transpose(0, 1).reshape(-1)
        exps.returnn_ext = self.returns_ext.transpose(0, 1).reshape(-1)

        # additional intrinsic stream required when using two-streams instead of one
        exps.advantage_int = self.advantages_int.transpose(0, 1).reshape(-1)
        exps.advantage_total = self.advantages_total.transpose(0,1).reshape(-1)
        exps.returnn_int = self.returns_int.transpose(0, 1).reshape(-1)

        # Preprocess experiences
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Add actual_int_coefs if used as input for Actor-Critic (for ngu case mandatory)
        shape = (self.num_frames_per_proc, self.num_procs)
        int_coefs = torch.zeros(*shape, device=self.device)
        for penv in range(self.num_procs):
            v = actual_int_coef[penv] if ((self.int_coef_type == 'ngu') or (self.int_coef_type == 'adaptive') or (self.int_coef_type == 'adaptive_1000')) else actual_int_coef
            int_coefs[:,penv] = v
        exps.int_coefs = int_coefs.transpose(0, 1).reshape(-1)

        ########################################################################
        # Log some values
        ########################################################################

        # weight of int coef to monitorize
        if self.int_coef_type=='static' or self.int_coef_type=='parametric':
            # only one value is assumed for all the penvs
            weight_int_coef = actual_int_coef
        elif self.int_coef_type=='adaptive' or self.int_coef_type=='adaptive_1000':
            # we store the avg among all the penvs weight coef
            weight_int_coef = actual_int_coef.mean(-1)
        else:
            # with ngu, take the agent with lowest value
            weight_int_coef = actual_int_coef[-1]


        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "return_int_per_episode": self.log_return_int[-keep:],
            "return_int_per_episode_norm": self.log_return_int_normalized[-keep:],
            "normalization_int_score": self.normalization_int_score,
            "episode_counter": self.episode_counter,
            "avg_return": mean(self.last_100return),
            "weight_int_coef": weight_int_coef,
            "predominance_ext_over_int": self.predominance_ext_over_int.mean().item(),
            "hist_ret_avg":hist_ret_avg,
            "time_forw_and_backw":total_time_fwandbw
        }

        # sobreescribe para redimensionar y empezar una nueva colecta
        self.log_done_counter = 0
        self.log_return_int = self.log_return_int[-self.num_procs:]
        self.log_return_int_normalized = self.log_return_int_normalized[-self.num_procs:]
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
