import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys

import utils
import torch
# for Actor-Critic
from model import ActorModel_RAPID, CriticModel_RAPID, ACModelRIDE

# ******************************************************************************
# Parse arguments
# ******************************************************************************

parser = argparse.ArgumentParser()

## General parameters

# Logs related
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=100,
                    help="number of updates between two saves (default: 100, 0 means no saving)")


# Environment dependant
parser.add_argument("--env", default='MiniGrid-MultiRoom-N7-S4-v0',
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--env-list", nargs="+" , default=[],
                    help="subset of files that we are going to use")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

## Parameters for intrinsic motivation
parser.add_argument("--intrinsic-motivation", type=float, default=0,
                    help="specify if we use intrinsic motivation (int_coef) to face sparse problems")
parser.add_argument("--im-type", default='counts',
                    help="specify if we use intrinsic motivation, which module/approach to use")
parser.add_argument("--int-coef-type", default='static',
                    help="specify how to decay the intrinsic coefficient")
parser.add_argument("--normalize-intrinsic-bonus", type=int, default=0,
                    help="boolean-int variable that set whether we want to normalize the intrinsic rewards or not")
parser.add_argument("--use-episodic-counts", type=int, default=0,
                    help="divide intrinsic rewards with the episodic counts for that given state")
parser.add_argument("--use-only-not-visited", type=int, default=0,
                    help="apply mask to reward only those states that have not been explored in the episode")



 ## Select algorithm and generic configuration params
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--separated-networks", type=int, default=0,
                    help="set if we use two different NN for actor and critic")

## Parameters for main algorithm

parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--nsteps", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

## GPU/CPU Configuration
parser.add_argument("--use-gpu", type=int, default=0,
                    help="Specify to use GPU as device to bootstrap the training")
parser.add_argument("--gpu-id", type=int, default=-1,
                    help="add a GRU to the model to handle text input")


args = parser.parse_args()

# ******************************************************************************
# AssertionError to ensure inconsistency problems
# ******************************************************************************
# LOGIC --> if it is true, no error thrown
assert ((args.env =='MiniGrid-NumpyMapFourRoomsPartialView-v0') and (len(args.env_list)>0)) \
        or args.env != 'MiniGrid-NumpyMapFourRoomsPartialView-v0', \
        'You have selected to use Pre-defined environments for training but no subfile specified'

assert (args.use_gpu==False) or (args.use_gpu and args.gpu_id != -1), \
        'Specify the device id to use GPU'



args.mem = args.recurrence > 1

# ******************************************************************************
# Set run dir
# ******************************************************************************

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# ******************************************************************************
# Load loggers and Tensorboard writer
# ******************************************************************************
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# ******************************************************************************
# Set seed for all randomness sources
# ******************************************************************************
utils.seed(args.seed)

# ******************************************************************************
# Set device
# ******************************************************************************
device = torch.device("cuda:"+str(args.gpu_id) if args.use_gpu else "cpu")
txt_logger.info(f"Device: {device}\n")


# ******************************************************************************
# Generate ENVIRONMENT DICT
# ******************************************************************************
# MiniGrid-MultiRoom-N7-S4-v0
env_dict = {}
env_list = [name + '.npy' for name in args.env_list] #add file extension
env_dict[args.env] = env_list
print('Env Dictionary:',env_dict)

# ******************************************************************************
# Load environments
# ******************************************************************************
envs = []
for i in range(args.procs):
    envs.append(utils.make_env(env_dict, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Define action_space
ACTION_SPACE = envs[0].action_space.n
txt_logger.info(f"ACTION_SPACE: {ACTION_SPACE}\n")

# ******************************************************************************
# Load training status
# ******************************************************************************

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# ******************************************************************************
# Load observations preprocessor
# ******************************************************************************
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")


# ******************************************************************************
# Load model
# ******************************************************************************
separated_networks = args.separated_networks
# Use 1 AC network or separated Actor and Critic
if separated_networks:
    actor = ActorModel_RAPID(obs_space, ACTION_SPACE, args.mem)
    critic = CriticModel_RAPID(obs_space, ACTION_SPACE, args.mem)
    actor.to(device)
    critic.to(device)
    if "model_state" in status:
        actor.load_state_dict(status["model_state"][0])
        critic.load_state_dict(status["model_state"][1])
    txt_logger.info("Models loaded\n")
    txt_logger.info("Actor: {}\n".format(actor))
    txt_logger.info("Critic: {}\n".format(critic))
    # save as tuple
    acmodel = (actor,critic)
    # calculate num of model params
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    total_params = actor_params + critic_params
    print('***PARAMS:\nActor {}\nCritic {}\nTotal {}'.format(actor_params,critic_params,total_params))
else:
    use_intcoefs = 1 if args.int_coef_type == 'ngu'else 0
    acmodel = ACModelRIDE(obs_space, ACTION_SPACE, use_intcoefs,args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    total_params = sum(p.numel() for p in acmodel.parameters())
    print('***PARAMS UNIQUE AC (RIDE):',total_params)
# ******************************************************************************
# Set Intrinsic Motivation
# ******************************************************************************
txt_logger.info("Intrinsic motivation:{}\n".format(args.im_type))
# ******************************************************************************
# Load algorithm
# ******************************************************************************

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, device, args.nsteps, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.nsteps, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                            separated_networks = args.separated_networks,
                            env_name=env_dict, num_actions = ACTION_SPACE,
                            int_coef=args.intrinsic_motivation,
                            normalize_int_rewards=args.normalize_intrinsic_bonus,
                            im_type=args.im_type, int_coef_type = args.int_coef_type,
                            use_episodic_counts = args.use_episodic_counts,
                            use_only_not_visited = args.use_only_not_visited,
                            total_num_frames = args.frames)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# ******************************************************************************
# Load optimizer
# ******************************************************************************

if "optimizer_state" in status:
    if separated_networks:
        algo.optimizer[0].load_state_dict(status["optimizer_state"][0])
        algo.optimizer[1].load_state_dict(status["optimizer_state"][1])
        txt_logger.info("Optimizer loaded\n")
    else:
        algo.optimizer.load_state_dict(status["optimizer_state"])
        txt_logger.info("Optimizer loaded\n")

# ******************************************************************************
# Train model
# ******************************************************************************

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        episodes = logs["episode_counter"]

        # intrinsic
        return_int_per_episode = utils.synthesize(logs["return_int_per_episode"])
        return_int__norm_per_episode = utils.synthesize(logs["return_int_per_episode_norm"])
        # extrinsic
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])


        # general values
        header = ["update", "frames", "FPS", "duration","episodes"]
        data = [update, num_frames, fps, duration, episodes]
        only_txt = [update, num_frames, fps, duration, episodes]

        # add beta coef
        header += ["weight_int_coef"]
        data += [logs["weight_int_coef"]]
        only_txt += [logs["weight_int_coef"]]

        # returns
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        only_txt += [rreturn_per_episode["mean"]]
        only_txt += [rreturn_per_episode["std"]]

        header += ["return_int_" + key for key in return_int_per_episode.keys()]
        data += return_int_per_episode.values()
        only_txt += [return_int_per_episode["mean"]]
        only_txt += [return_int_per_episode["std"]]

        # avg 100 episodes
        header += ["avg_return"]
        data += [logs["avg_return"]]
        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm", "grad_norm_critic"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["grad_norm_critic"]]
        only_txt += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["grad_norm_critic"]]

        header+= ["hist_ret_avg","normalization_int_score","predominance_ext_over_int"]
        data += [logs["hist_ret_avg"],logs["normalization_int_score"],logs["predominance_ext_over_int"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | Eps {} | β:{:.5f} |rR:μσ {:.2f} {:.2f} | rRi:μσ {:.2f} {:.2f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇p {:.3f} | ∇c {:.3f}"
            .format(*only_txt))
        # txt_logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | H {:.3f} | V {:.3f} | Ve {:.3f} | Vi {:.3f}  | pL {:.3f} | vL {:.3f} | vLe {:.3f} | vLi {:.3f} | ∇p {:.3f} | ∇c {:.3f}"
        #     .format(*data))
        # with normalized intrinsic return
        # txt_logger.info(
        #     "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | rR_intNORM:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | H {:.3f} | V {:.3f} | Ve {:.3f} | Vi {:.3f}  | pL {:.3f} | vL {:.3f} | veL {:.3f} | viL {:.3f} | ∇ {:.3f}"
        #     .format(*data))

        # THIS ONLY NECESSARY IF MODIFICATED RESHAPE REWARD
        # header += ["return_" + key for key in return_per_episode.keys()]
        # data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        if separated_networks:
            acmodel_weights = (acmodel[0].state_dict(),
                                acmodel[1].state_dict())
            optimizer_state = (algo.optimizer[0].state_dict(),
                                algo.optimizer[1].state_dict())
        else:
            acmodel_weights = acmodel.state_dict()
            optimizer_state = algo.optimizer.state_dict()
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel_weights, "optimizer_state": optimizer_state}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
