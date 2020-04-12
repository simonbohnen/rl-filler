from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from filler import FillerState, get_random_regular_board

num_iterations = 2000000

initial_collect_steps = 1000
collect_steps_per_iteration = 10
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-5
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
        self._episode_ended = False
        self.state = None
        self.reset_state()
        self.total_score = 1
        self.rewards = []

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if self.state.move_count > 200:
            print("Canceling game due to stalling.")
            reward = -1
            self._episode_ended = True
        else:
            new_owned_count = self.state.move(action)
            if self.state.last_move_illegal:
                reward = -1
            else:
                reward = new_owned_count
            self.total_score += new_owned_count
            if not self.state.is_final_state:
                self.state.move(random.randint(0, 5))
            if self.state.is_final_state:
                self._episode_ended = True
                if self.total_score > 28:
                    reward += 10
                # elif self.total_score < 28:
                #     reward -= 10
                # print(str_board(self.state.board))
                # print(self.total_score)

        self.rewards.append(reward)
        if self._episode_ended:
            return ts.termination(self.state.board, reward)
        else:
            # TODO was fürn discount?
            return ts.transition(self.state.board, reward, 0.7)

    def reset_state(self):
        self.total_score = 1
        self.rewards = []
        first_player = 1  # random.randint(1, 2)
        self.state = FillerState(get_random_regular_board(), first_player)
        if first_player == 2:
            self.state.move(random.randint(0, 5))  # do_standard_move()

    def _reset(self):
        self._episode_ended = False
        self.reset_state()
        return ts.restart(self.state.board)

    def get_info(self):
        # No idea what I should do here
        pass


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    score_sum = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        if episode_return < -102:
            print("How's that possible? negative reward count: {0}"
                  .format(environment.pyenv.envs[0].rewards.count(-10)))
        total_return += episode_return
        score_sum += environment.pyenv.envs[0].total_score

    print("Avg score: %f" % (score_sum / num_episodes))
    average_return = total_return / num_episodes
    return np.float(average_return)


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


if __name__ == "__main__":
    valenv = FillerEnvironment()
    utils.validate_py_environment(valenv)
    print("\nEverything ok.")

    tf.compat.v1.enable_v2_behavior()
    train_env = tf_py_environment.TFPyEnvironment(FillerEnvironment())
    eval_env = tf_py_environment.TFPyEnvironment(FillerEnvironment())
    fc_layer_params = (50,)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec())
    # fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    # noinspection PyTypeChecker
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print("Initial average return: {0}".format(avg_return))
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
