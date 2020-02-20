from typing import Optional
#=======================
from .policy_approximator import Policy
from .environment_sensor_map import Environment, Sensor, CognitiveMap
from .state_action import Action, Observation, BeliefState
#=======================

class Agent:
    def __init__(
        self, 
        sensor: Sensor, 
        policy: Policy, 
        cognitive_map: CognitiveMap,
        belief_state:Optional[BeliefState] = None,
        observation:Optional[Observation] = None,
    ) -> None:
        self.sensor = sensor
        self.policy = policy 
        self.cognitive_map = cognitive_map
        self.belief_state = belief_state
        
    def select_action(
        self, 
        state:Environment,
        last_action:Action,
    ) -> Action:

        self.observation = self.sensor.get_observation(state)

        self.belief_state = self.cognitive_map.get_belief_state(
            previous_action = last_action,
            previous_belief_state = self.belief_state,
            observation = self.observation,
        )

        return self.policy.get_action(
            belief_state = self.belief_state 
        )

class RLExperiment:
    def __init__(
        self, 
        environment:Environment, 
        agent:Agent, 
        action:Optional[Action] = None, 
        verbose:bool=True
    ) -> None:

        self.agent = agent 
        self.environment = environment
        self.verbose = verbose
        self.action = action
    
    def run(self, number_of_steps:int = 100) -> None:
        self._iterate(number_of_steps)

        if self.verbose: 
            print(f"n iterations = {number_of_steps}")
            print(self.agent.policy)
            self.plot_results()

    def _step(self) -> None:
        self.action = self.agent.select_action(
            state = self.environment,
            last_action = self.action,
        )
        self.environment.update_state(self.action)
        if self.verbose:
            print(self.agent.observation)
            print(self.agent.belief_state)
            print(self.action)
            print("-"*15)
    
    def _iterate(self, number_of_steps:int) -> None:
        for _ in range(number_of_steps):
            self._step()

    def plot_results(self) -> None:
        pass 
        #TODO 
    
