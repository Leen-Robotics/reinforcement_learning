from typing import Optional
from random import choice, sample
#=======================
from .policy_approximator import Policy, PolicyFunction, ValueFunction, load_in_action_space
from .state_action import Action, BeliefState
#from .machine_learning_models.extreme_learning_machine import ExtremeLearningMachine
#=======================

class RandomExploration(Policy):
    def __str__(self) -> str:
        return __class__.__name__

    def get_action(self, belief_state:BeliefState) -> Action:
        return choice(self.action_space)   


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

    def infer_reward(belief_state:BeliefState) -> float:
        raise NotImplementedError
    
    def _discounted_reward(self, belief_state:BeliefState, depth:int=1) -> float:
        discount_factor = 1/(depth+1)
        return self.infer_reward(belief_state) * discount_factor 

    def _planning_algorithm(
        self, 
        belief_state:BeliefState, 
        accumulated_reward:float = 0., 
        current_depth:int = 0
    ) -> float:
        scores_of_sampled_trajectories = []
        for next_action in sample(self.action_space, k=self.sample_size):
            next_belief_state = self.infer_belief_state(
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
    
    def infer_state_value(self, belief_state:BeliefState) -> float:
        return self._planning_algorithm(belief_state)

#class ELMPolicyFunction(PolicyFunction, ExtremeLearningMachine):
    #TODO