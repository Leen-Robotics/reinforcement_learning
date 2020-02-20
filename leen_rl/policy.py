#=======================
from .state_action import Action, BeliefState, load_in_action_space
from .approximators import π, φ, V, Q
#=======================

class Policy:
    def __init__(self,action_space_path: str) -> None:
        self.action_space = load_in_action_space(action_space_path)
    
    def get_action(self, belief_state:BeliefState) -> Action:
        raise NotImplementedError

class PolicyFunction(Policy, π):
    def __str__(self) -> str:
        return __class__.__name__

    def get_action(self, belief_state:BeliefState) -> Action:
        return self.infer_action(belief_state)

class QFunction(Policy, Q):
    def __str__(self) -> str:
        return __class__.__name__

    def get_action(self, belief_state:BeliefState) -> Action:
        return max(
            self.action_space,
            key = lambda action: self.infer_action_value(
                action = action,
                belief_state = belief_state,
            )
        )          
        
class ValueFunction(Policy, V, φ):
    def __str__(self) -> str:
        return __class__.__name__

    def get_action(self, belief_state:BeliefState) -> Action:
        return max(
            self.action_space,
            key = lambda action: self.infer_state_value(
                belief_state = self.infer_belief_state(
                    action = action,
                    belief_state = belief_state
                )
            )
        )