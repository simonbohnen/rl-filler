from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import filler
from filler import FillerState, get_random_regular_board
import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

num_iterations = 20000

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 200

num_eval_episodes = 10
eval_interval = 1000


class FillerEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8, 7), dtype=np.int32, minimum=0, maximum=5, name='observation')
        self._state = 0
        self._episode_ended = False
        self.state = FillerState(get_random_regular_board(), 1)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        pass

    def _reset(self):
        self._episode_ended = False
        self.state = FillerState(get_random_regular_board(), 1)
        return ts.restart(self.state.board)
