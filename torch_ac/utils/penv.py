from multiprocessing import Process, Pipe
import gym
import numpy as np

def worker(conn, env, env_dict):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                env_key = list(env_dict.keys())[0]
                if env_key == 'MiniGrid-NumpyMapFourRoomsPartialView-v0':
                    possible_envs = list(env_dict.values())[0]
                    selected_env = np.random.choice(possible_envs)
                    env = gym.make(env_key,numpyFile='numpyworldfiles/' + selected_env,max_steps=100)
                # reset
                obs = env.reset()

            info = env.agent_pos
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            env_key = list(env_dict.keys())[0]
            if env_key == 'MiniGrid-NumpyMapFourRoomsPartialView-v0':
                possible_envs = list(env_dict.values())[0]
                selected_env = np.random.choice(possible_envs)
                env = gym.make(env_key,numpyFile='numpyworldfiles/' + selected_env,max_steps=100)
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, env_dict):
        assert len(envs) >= 1, "No environment given."
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env, env_dict))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        info = self.envs[0].agent_pos
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
