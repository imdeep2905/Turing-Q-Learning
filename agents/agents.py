import logging
import time
import csv
import os
import functools
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from pathlib import Path
from tensorflow.keras.models import Sequential
from envs.utils import get_avg_reward, create_eval_video

tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()


class DQNAgentBase:
    def __init__(
        self,
        envs,
        discount_rate,
        optimizer,
        loss_fn,
        common_model,
        batch_size=32,
        identifier="DQN Agent Base Class",
    ):
        self.envs = envs
        self.discount_rate = discount_rate
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.common_model = common_model
        self.batch_size = batch_size
        self.identifier = identifier
        logging.debug(
            f"{self.__class__.__name__}, {self.identifier}: Agent created."
        )

    @tf.function
    def _predict(self, index, _input):
        output = self.envs[index].in_interactor.call(inputs=_input)
        output = self.common_model.call(output)
        output = self.envs[index].out_interactor.call(inputs=output)
        return output

    def _get_agent(self, index):
        return functools.partial(self._predict, index)

    def get_model(self, index):
        return Sequential(
            [
                self.envs[index].in_interactor,
                self.common_model,
                self.envs[index].out_interactor,
            ]
        )

    def _save_config(
        self,
        num_iterations,
        steps_per_iter,
        log_dir,
        eval_episodes,
        log_interval,
        video_save_interval,
    ):
        with open(os.path.join(log_dir, "config.txt"), "w") as file:
            file.write("Environments:\n")
            for env in self.envs:
                file.write(f"\t{env.name}\n")
            file.write(f"Discount Rate: {self.discount_rate}\n")
            file.write("Optimizer:\n")
            file.write(f"\t{self.optimizer.__class__.__name__}")
            for key, val in self.optimizer.get_config().items():
                file.write(f"\t{key}: {val}\n")
            file.write("Loss Function:\n")
            file.write(f"\t{self.loss_fn.__class__.__name__}\n")
            for key, val in self.loss_fn.get_config().items():
                file.write(f"\t{key}: {val}\n")
            file.write(f"num_iterations: {num_iterations}\n")
            file.write(f"steps_per_iter: {steps_per_iter}\n")
            file.write(f"eval_episodes: {eval_episodes}\n")
            file.write(f"log_interval: {log_interval}\n")
            file.write(f"video_save_interval: {video_save_interval}\n")

    @abstractmethod
    def _step(self, pseudo_learning_rate):
        raise NotImplementedError

    def train(
        self,
        num_iterations,
        pseudo_learning_rate,
        steps_per_iter=4,
        log_dir=None,
        eval_episodes=5,
        log_interval=1000,
        video_save_interval=5000,
    ):
        if log_dir is None:
            log_dir = Path(
                os.path.join("runs", time.strftime("%d_%m_%Y-%H_%M_%S"))
            )
            os.makedirs(log_dir)
        logging.info(f"{ __name__}: Saving logs in {log_dir}.")
        with open(os.path.join(log_dir, "logs.csv"), "a+", newline="") as file:
            writer = csv.writer(file)
            row = ["iter_no", "avg_loss"]
            for env in self.envs:
                row.append("avg_reward_" + env.name)
            writer.writerow(row)

        self._save_config(
            num_iterations,
            steps_per_iter,
            log_dir,
            eval_episodes,
            log_interval,
            video_save_interval,
        )

        for iter in range(1, num_iterations + 1):
            for i in range(len(self.envs)):
                for _ in range(steps_per_iter):
                    self.envs[i]._step_driver(
                        np.argmax(
                            self._predict(
                                i,
                                self.envs[i]
                                .env.current_time_step()
                                .observation,
                            )
                        )
                    )

            loss = self._step(pseudo_learning_rate)

            print(f"Current Iteration: {iter}")

            if iter % log_interval == 0:
                print(f"\tAverage loss: {loss}.")
                row = [iter, loss.numpy()[0]]
                for env_num in range(len(self.envs)):
                    avg_reward = get_avg_reward(
                        self.envs[env_num],
                        self._get_agent(env_num),
                        eval_episodes,
                    )
                    row.append(avg_reward)
                    print(
                        f"\t{self.envs[env_num].name}: Average reward over"
                        f" {eval_episodes} episodes: {avg_reward}."
                    )

                with open(
                    os.path.join(log_dir, "logs.csv"), "a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

                self.save(log_dir, iter)

            if iter % video_save_interval == 0:
                for env_num in range(len(self.envs)):
                    create_eval_video(
                        os.path.join(
                            log_dir,
                            self.envs[env_num].name + f"_video_iter_{iter}",
                        ),
                        self.envs[env_num],
                        self._get_agent(env_num),
                        eval_episodes,
                    )

    def save(self, dir, iter):
        self.common_model.save_weights(
            os.path.join(dir, f"dnc_iter_{iter}.h5")
        )
        for i in range(len(self.envs)):
            name = self.envs[i].name
            name = name.replace("|", "-")
            name = name.replace(" ", "_")
            self.envs[i].in_interactor.save(
                os.path.join(dir, f"{name}_in_iter_{iter}.h5")
            )
            self.envs[i].out_interactor.save(
                os.path.join(dir, f"{name}_out_iter_{iter}.h5")
            )
        logging.info(
            f"All models have been saved successfully for iter : {iter}"
        )

    def load(self, dir, iter):
        raise NotImplementedError


class DQNAgentSummedLoss(DQNAgentBase):
    @tf.function
    def _step(self, pseudo_learning_rate):
        with tf.GradientTape(persistent=True) as tape:

            total_loss = tf.zeros((1, 1))

            for env_num in range(len(self.envs)):
                experiences = self.envs[env_num].get_batch_from_replay_buffer(
                    batch_size=self.batch_size
                )

                states, actions, rewards, next_states, dones = experiences[0]

                next_Q_values = self._predict(env_num, next_states)
                max_next_Q_values = tf.math.reduce_max(next_Q_values)
                target_Q_values = (
                    tf.reshape(rewards, (1, self.batch_size))
                    + tf.math.subtract(
                        tf.ones((1, self.batch_size)),
                        tf.reshape(
                            tf.cast(dones, tf.float32), (1, self.batch_size)
                        ),
                    )
                    * self.discount_rate
                    * max_next_Q_values
                )

                target_Q_values = tf.reshape(target_Q_values, (-1, 1))

                mask = tf.one_hot(
                    actions, self.envs[env_num].number_of_actions
                )

                all_Q_values = self._predict(env_num, states)

                Q_values = tf.reduce_sum(
                    all_Q_values * mask, axis=1, keepdims=True
                )

                loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

                total_loss = tf.math.add(total_loss, loss)

                in_grad = tape.gradient(
                    loss, self.envs[env_num].in_interactor.trainable_variables
                )

                self.optimizer.apply_gradients(
                    zip(
                        in_grad,
                        self.envs[env_num].in_interactor.trainable_variables,
                    )
                )

                out_grad = tape.gradient(
                    loss, self.envs[env_num].out_interactor.trainable_variables
                )

                self.optimizer.apply_gradients(
                    zip(
                        out_grad,
                        self.envs[env_num].out_interactor.trainable_variables,
                    )
                )

            common_model_grads = tape.gradient(
                total_loss, self.common_model.trainable_variables
            )

            for i in range(len(common_model_grads)):
                common_model_grads[i] = tf.math.scalar_mul(
                    pseudo_learning_rate, common_model_grads[i]
                )

            self.optimizer.apply_gradients(
                zip(
                    common_model_grads,
                    self.common_model.trainable_variables,
                )
            )

        return total_loss / len(self.envs)
