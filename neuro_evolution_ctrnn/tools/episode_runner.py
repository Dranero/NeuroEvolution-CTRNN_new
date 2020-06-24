import gym
from tools.helper import set_random_seeds, output_to_action
from tools.configurations import EpisodeRunnerCfg
import logging
from tools.dask_handler import get_current_worker
from tools.env_handler import EnvHandler


class EpisodeRunner(object):
    def __init__(self, conf: EpisodeRunnerCfg, brain_conf: object, action_space, brain_class, input_space,
                 output_space, env_template):
        self.conf = conf
        self.action_space = action_space
        self.brain_class = brain_class
        self.brain_conf = brain_conf
        self.input_space = input_space
        self.output_space = output_space
        self.env_id = env_template.spec.id
        self.env_handler = EnvHandler(self.conf)


    def eval_fitness(self, individual, seed):
        if self.conf.reuse_env:
            try:
                env = get_current_worker().env
            except:
                if hasattr(self, "env"):
                    env = self.env
                else:
                    self.env = env =  self.env_handler.make_env(self.env_id)
        else:
            env = self.env_handler.make_env(self.env_id)
        set_random_seeds(seed, env)
        fitness_total = 0
        for i in range(self.conf.number_fitness_runs):
            fitness_current = 0
            brain = self.brain_class(self.input_space, self.output_space, individual,
                                     self.brain_conf)
            ob = env.reset()
            done = False
            while not done:
                brain_output = brain.step(ob)
                action = output_to_action(brain_output, self.output_space)
                ob, rew, done, info = env.step(action)
                fitness_current += rew
            fitness_total += fitness_current

        return fitness_total / self.conf.number_fitness_runs, env.get_compressed_behavior(),
