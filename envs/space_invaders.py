from tf_agents.environments import suite_gym
from envs.env_base import BaseEnvWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class SpaceInvader(BaseEnvWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(
            TFPyEnvironment(suite_gym.load("SpaceInvaders-v0")),
            "SpaceInvaders V0 | No Preprocessing",
            *args,
            **kwargs
        )
