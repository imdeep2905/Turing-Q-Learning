import logging
import numpy as np
import tensorflow as tf
from tf_agents.specs import TensorSpec
from envs.utils import Epsilon, TFUniformReplayBufferWrapper


class BaseEnvWrapper:
    def __init__(
        self,
        env,
        eval_env,
        name,
        in_interactor,
        out_interactor,
        replay_buffer_size=10_000,
        replay_buffer_initialization_size=1000,
        epsilon_initial_value=1.0,
        epsilon_end_value=0.1,
        epsilon_decay_steps=10_000,
        epsilon_power=1,
    ):

        self.name = name

        self.env = env
        self.eval_env = eval_env

        self.number_of_actions = self.env.action_spec().maximum + 1

        self._epislon = Epsilon(
            initial_value=epsilon_initial_value,
            end_value=epsilon_end_value,
            decay_steps=epsilon_decay_steps,
            power=epsilon_power,
            identifier=f"Epsilon ({self.name})",
        )

        self._replay_buffer_size = replay_buffer_size
        self._replay_buffer_batch_size = self.env.batch_size
        self._replay_buffer = TFUniformReplayBufferWrapper(
            self._replay_buffer_size,
            self._replay_buffer_batch_size,
            (
                self.env.observation_spec(),
                TensorSpec(shape=(), dtype=tf.int32),
                TensorSpec(shape=(), dtype=tf.float32),
                self.env.observation_spec(),
                TensorSpec(shape=(), dtype=tf.bool),
            ),
            self.name,
        )

        self.in_interactor = in_interactor
        self.out_interactor = out_interactor

        logging.debug(
            f"{self.__class__.__name__}, {self.name}: Object created."
        )

        self.init_replay_buffer(replay_buffer_initialization_size)

    def init_replay_buffer(self, size):
        logging.info(
            f"Initializing replay buffer of {self.name} with size {size}."
        )
        for _ in range(size):
            cur_obs = self.env.current_time_step()
            action = np.random.randint(0, self.number_of_actions)
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            next_obs = self.env.step(action)
            self._replay_buffer.add_batch(
                (
                    cur_obs.observation,
                    tf.expand_dims(action, axis=0),
                    next_obs.reward,
                    next_obs.observation,
                    next_obs.is_last(),
                )
            )
            if next_obs.is_last():
                self.env.reset()

    def get_batch_from_replay_buffer(self, batch_size):
        return self._replay_buffer.sample_batch(batch_size)

    def _step_driver(self, action):
        cur_obs = self.env.current_time_step()
        if np.random.rand() > self._epislon.epsilon:
            action = tf.convert_to_tensor(action, dtype=tf.int32)
        else:
            action = tf.convert_to_tensor(
                np.random.randint(0, self.number_of_actions),
                dtype=tf.int32,
            )
        next_obs = self.env.step(action)
        self._replay_buffer.add_batch(
            (
                cur_obs.observation,
                tf.expand_dims(action, axis=0),
                next_obs.reward,
                next_obs.observation,
                next_obs.is_last(),
            )
        )
        if next_obs.is_last():
            self.env.reset()
