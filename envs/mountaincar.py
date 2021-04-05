import gym
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from envs.env_base import BaseEnvWrapper


class MountainCarV0(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("MountainCar-v0")),
            gym.make("MountainCar-v0"),
            "MountainCar v0 | No Preprocessing",
            *args,
            **kwargs
        )


class MountainCarContinuousV0(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("MountainCarContinuous-v0")),
            gym.make("MountainCarContinuous-v0"),
            "MountainCarContinuous v0 | No Preprocessing",
            *args,
            **kwargs
        )
