#=======================
from .state_action import BeliefState, Action
#=======================
class Approximator:
    def learn(self) -> None:
        raise NotImplementedError 

class π(Approximator):
    def infer_action(self, belief_state:BeliefState) -> Action:
        raise NotImplementedError

class φ(Approximator):
    def infer_belief_state(self, belief_state:BeliefState, action:Action) -> BeliefState:
        raise NotImplementedError

class V(Approximator):
    def infer_state_value(self, action:Action) -> float:
        raise NotImplementedError

class Q(Approximator):
    def infer_action_value(self, belief_state:BeliefState, action:Action) -> float:
        raise NotImplementedError

class R(Approximator):
    def infer_reward(self, belief_state:BeliefState) -> float:
        raise NotImplementedError