from typing import Optional
#=======================
from .state_action import BeliefState, Observation, Action
#=======================

class Environment:
    def __init__(self) -> None:
        self.initialise_state()

    def initialise_state(self) -> None:
        raise NotImplementedError 

    def update_state(self, action:Action) -> None:
        raise NotImplementedError

class Sensor:
    def get_observation(self, state:Environment) -> Observation:
        raise NotImplementedError

class CognitiveMap:    
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState] = None,
        previous_action:Optional[Action] = None
    ) -> BeliefState:
        raise NotImplementedError