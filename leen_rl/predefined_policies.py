from typing import Optional
from random import choice
#=======================
from .policy import Policy
from .state_action import Action, BeliefState
#=======================

class RandomExploration(Policy):
    def __str__(self) -> str:
        return __class__.__name__

    def get_action(self, belief_state:BeliefState) -> Action:
        return choice(self.action_space)       