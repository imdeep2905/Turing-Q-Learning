from tf_agents.environments import suite_gym
from envs.env_base import BaseEnvWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class BreakoutV0(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("Breakout-v0")),
            "Breakout v0 | No Preprocessing",
            *args,
            **kwargs
        )


class BreakoutRamV0(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("Breakout-ram-v0")),
            "Breakout-ram v0 | No Preprocessing",
            *args,
            **kwargs
        )
