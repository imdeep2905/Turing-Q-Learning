import tensorflow as tf
import numpy as np
import logging
import imageio
from tf_agents.replay_buffers.tf_uniform_replay_buffer import (
    TFUniformReplayBuffer,
)

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


class TFUniformReplayBufferWrapper:
    def __init__(self, max_length, batch_size, data_spec, env_name):
        self._buffer = TFUniformReplayBuffer(
            data_spec=data_spec, batch_size=batch_size, max_length=max_length
        )
        self._env_name = env_name
        logging.debug(
            f"{self.__class__.__name__ }({self._env_name}): Replay"
            " buffer created."
        )

    def sample_batch(self, batch_size):
        logging.debug(
            f"{self.__class__.__name__ }({self._env_name}): Sampling batch"
            f" with batch_size: {batch_size}."
        )
        return self._buffer.get_next(sample_batch_size=batch_size)

    def add_batch(self, batch):
        logging.debug(
            f"{self.__class__.__name__ }({self._env_name}): Adding batch"
            f" with batch_size: {len(batch)}."
        )
        self._buffer.add_batch(batch)


class Epsilon:
    def __init__(
        self,
        initial_value=1.0,
        end_value=0.01,
        decay_steps=10_000,
        power=1,
        identifier=None,
    ):
        self._initial_value = initial_value
        self._end_value = end_value
        self._epsilon = initial_value
        self._step_count = 1
        self._decay_steps = decay_steps
        self._power = power
        if identifier is None:
            self._identifier = ""
        else:
            self._identifier = identifier
        logging.debug(
            f"{self.__class__.__name__}, {self._identifier}: Epsilon object"
            " created."
        )

    @property
    def epsilon(self):
        if self._step_count <= self._decay_steps:
            self._epsilon = (
                (self._initial_value - self._end_value)
                * (1 - self._step_count / self._decay_steps) ** (self._power)
            ) + self._end_value
            self._step_count += 1
        return self._epsilon


def create_eval_video(filename, _env, model, num_episodes, fps=30):
    env = _env.eval_env

    logging.debug(
        f"{__name__}: Creating agent video. (env = {_env.name},"
        f" episodes = {num_episodes})."
    )
    filename += ".mp4"
    filename = filename.replace("|", "-")
    filename = filename.replace(" ", "_")

    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(1, num_episodes + 1):
            logging.debug(f"{__name__}: {_env.name}, Video Episode: {i}.")
            current_observation = env.reset()
            video.append_data(env.render(mode="rgb_array"))
            done = False
            while not done:
                action = np.argmax(
                    model(
                        tf.convert_to_tensor(
                            np.expand_dims(current_observation, axis=0)
                        )
                    )
                )
                current_observation, _, done, _ = env.step(action)
                video.append_data(env.render(mode="rgb_array"))
        env.close()

    logging.debug(
        f"{__name__}: Video successfully saved in the file: {filename}."
    )


def get_rewards(_env, model, num_episodes):
    env = _env.eval_env
    logging.debug(
        f"{__name__}: Calculating rewards. (env = {_env.name},"
        f" episodes = {num_episodes})."
    )
    rewards = []
    for i in range(1, num_episodes + 1):
        current_observation = env.reset()
        current_reward = 0.0
        done = False
        while not done:
            action = np.argmax(
                model(
                    tf.convert_to_tensor(
                        np.expand_dims(current_observation, axis=0)
                    )
                )
            )
            current_observation, _reward, done, _ = env.step(action)
            current_reward += _reward
        logging.debug(
            f"{__name__}: {_env.name}, Episode: {i} Reward: {current_reward}."
        )
        rewards.append(current_reward)
    return rewards


def get_avg_reward(_env, model, num_episodes):
    avg_reward = sum(get_rewards(_env, model, num_episodes)) / num_episodes
    logging.debug(
        f"{__name__}: {_env.name}, Avg Reward: {avg_reward},"
        f" num_episodes: {num_episodes}."
    )
    return avg_reward
