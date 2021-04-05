import gym
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from envs.env_base import BaseEnvWrapper


class CartPoleV0(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("CartPole-v0")),
            gym.make("CartPole-v0"),
            "CartPole V0 | No Preprocessing",
            *args,
            **kwargs
        )


class CartPoleV1(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("CartPole-v1")),
            gym.make("CartPole-v1"),
            "CartPole V1 | No Preprocessing",
            *args,
            **kwargs
        )
