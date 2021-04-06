from tf_agents.environments import suite_gym
from envs.env_base import BaseEnvWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
import gym


class AcrobotV1(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("Acrobot-v1")),
            gym.make("Acrobot-v1"),
            "Acrobot V1 | No Preprocessing",
            *args,
            **kwargs
        )
