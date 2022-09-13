import argparse
import numpy

import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")

# Environment related
parser.add_argument("--env", default='MiniGrid-NumpyMapFourRoomsPartialView-v0',
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--env-list", nargs="+" , default=[],
                    help="subset of files that we are going to use")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")


parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--separated-networks", type=int, default=0,
                    help="set if we use two different NN for actor and critic")

args = parser.parse_args()


assert ((args.env =='MiniGrid-NumpyMapFourRoomsPartialView-v0') and (len(args.env_list)>0)) \
        or args.env != 'MiniGrid-NumpyMapFourRoomsPartialView-v0', \
        'You have selected to use Pre-defined environments for training but no subfile specified'
# Generate ENVIRONMENT DICT
env_dict = {}
env_list = [name + '.npy' for name in args.env_list] #add file extension
env_dict[args.env] = env_list
print('Env Dictionary:',env_dict)
# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

print(f"Device: {device}\n")

# Load environment

env = utils.make_env(env_dict, args.seed)
for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent

ACTION_SPACE = env.action_space.n

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, ACTION_SPACE, model_dir,
                    argmax=args.argmax, use_memory=args.memory, use_text=args.text,
                    separated_networks=args.separated_networks)
print("Agent loaded\n")

# Run the agent

if args.gif:
   from array2gif import write_gif
   frames = []

# Create a window to view the environment
env.render('human')

for episode in range(args.episodes):
    obs = env.reset()
    steps= 0
    while True:
        steps += 1
        env.render('human')
        if args.gif:
            frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            print('Num of steps:',steps)
            break

    if env.window.closed:
        break

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
    print("Done.")
