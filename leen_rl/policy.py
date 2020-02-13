from typing import Optional
from random import sample
#=======================
from .state_action import Action, BeliefState, load_in_action_space
#=======================

class Policy:
    def __init__(self,action_space_path: str) -> None:
        self.action_space = load_in_action_space(action_space_path)
    
    def get_action(self, belief_state:BeliefState) -> Action:
        raise NotImplementedError

class PolicyFunction(Policy):
    def __str__(self) -> str:
        return __class__.__name__

    def policy_function(self, belief_state:BeliefState) -> Action:
        raise NotImplementedError

    def get_action(self, belief_state:BeliefState) -> Action:
        return self.policy_function(belief_state)

class QFunction(Policy):
    def __str__(self) -> str:
        return __class__.__name__

    def q_function(self, belief_state:BeliefState, action:Action) -> float:
        raise NotImplementedError

    def get_action(self, belief_state:BeliefState) -> Action:
        return max(
            self.action_space,
            key = lambda action: self.q_function(
                action = action,
                belief_state = belief_state,
            )
        )          
        
class ValueFunction(Policy):
    def __str__(self) -> str:
        return __class__.__name__

    def state_transition_function(self, action:Action, belief_state:BeliefState) -> BeliefState:
        raise NotImplementedError

    def value_function(self, belief_state:BeliefState) -> float:
        raise NotImplementedError

    def get_action(self, belief_state:BeliefState) -> Action:
        return max(
            self.action_space,
            key = lambda action: self.value_function(
                belief_state = self.state_transition_function(
                    action = action,
                    belief_state = belief_state
                )
            )
        )

class GoalPlanning(ValueFunction):
    def __init__(
        self,
        action_space_path: str,
        max_depth:int = 1,
        sample_size:Optional[int] = None,
    ) -> None:
        self.action_space = load_in_action_space(action_space_path)
        self.max_depth = max_depth
        self.sample_size = sample_size if sample_size else len(self.action_space)

    def __str__(self) -> str:
        return __class__.__name__

    def _discounted_reward(self, belief_state:BeliefState, depth:int=1) -> float:
        discount_factor = 1/(depth+1)
        return self.reward_function(belief_state) * discount_factor 

    def reward_function(self, belief_state:BeliefState) -> float:
        raise NotImplementedError

    def _planning_algorithm(
        self, 
        belief_state:BeliefState, 
        accumulated_reward:float = 0., 
        current_depth:int = 0
    ) -> float:
        scores_of_sampled_trajectories = []
        for next_action in sample(self.action_space, k=self.sample_size):
            next_belief_state = self.state_transition_function(
                action = next_action,
                belief_state = belief_state
            )
            accumulated_reward += self._discounted_reward(
                belief_state=next_belief_state,
                depth=current_depth,
            )
            if current_depth == self.max_depth:
                return accumulated_reward
            future_score = self._planning_algorithm(
                belief_state=next_belief_state,
                accumulated_reward=accumulated_reward,
                current_depth=current_depth+1
            )
            scores_of_sampled_trajectories.append(future_score)
        return sum(scores_of_sampled_trajectories) / self.sample_size
    
    def value_function(self, belief_state:BeliefState) -> float:
        return self._planning_algorithm(belief_state)