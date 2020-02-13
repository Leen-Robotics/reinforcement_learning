from gym import make
from typing import Optional, Tuple, List
from numpy import asarray
from PIL import Image
#=======================
from leen_rl import Agent
from leen_rl import RLExperiment
from leen_rl import Environment
from leen_rl import Sensor
from leen_rl import CognitiveMap
from leen_rl import QFunction
from leen_rl import Action
from leen_rl import Observation
from leen_rl import BeliefState
#=======================

class Racetrack(Environment):
    def initialise_state(self) -> None:
        self.environment = make("CarRacing-v0")
        self.pixels = self.environment.reset()
        self.external_reward = 0
        self.tick = 0
        self.environment.render()

    def update_state(self, action:Action) -> None:
        self.pixels,self.external_reward, _, _ = self.environment.step(
            action = [action.steer, action.gas, action.brake]
        )
        self.environment.render()
        self.tick += 1

class DriverView(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            pixels = state.pixels,
            reward = state.external_reward,
            tick = state.tick,
        )

class DriverMind(CognitiveMap):
    def _reduce_size_of_pixels(
        self,
        pixels:List[List[List[float]]],
        reduced_size:Tuple[int,int] = (28,28)
    ) -> List[List[List[float]]]:

        return asarray(
            Image.fromarray(pixels).resize(reduced_size)
        )

    def _flatten_rgb_channels(
        self,
        pixels:List[List[List[float]]],
        rgb_channel_weights:Tuple[float,float,float] = (.2989,.587,.114),
    ) -> List[List[float]]:
        return sum(
            rgb_channel_weights[index]*pixels[:,:,index] for index in range(3)
        )
    
    def _normalise_pixels(self,pixels:List[List[float]]) -> List[List[float]]:
        return pixels * 1/255
    
    def _feature_engineer_map(self,pixels:List[List[List[float]]]) -> List[List[float]]:
        return self._normalise_pixels(
            pixels = self._flatten_rgb_channels(
                pixels = self._reduce_size_of_pixels(pixels=pixels)
            )
        )

    def _feature_engineer_acceleration(self, action:Optional[Action]) -> float:
        return max(0,action.gas - action.brake) if action else 0.

    def _feature_engineer_heading(
        self,
        action:Optional[Action],
        belief_state:Optional[BeliefState]
    ) -> float:
        return (
            belief_state.heading if belief_state else 0
        ) + (
            action.steer if action else 0
        ) 

    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState]=None,
        previous_action:Optional[Action]=None
    ) -> BeliefState:

        previous_acceleration = self._feature_engineer_acceleration(action=previous_action)
        
        return BeliefState(
            reward = observation.reward,
            reduced_pixels = self._feature_engineer_map(pixels=observation.pixels[:-15,:,:]),
            previous_acceleration = previous_acceleration,
            previous_turn = previous_action.steer if previous_action else 0,
            velocity = previous_belief_state.velocity + previous_acceleration if previous_belief_state else 0,
            heading = self._feature_engineer_heading(action=previous_action,belief_state=previous_belief_state),
        )

class DriverDecisions(QFunction):
   def q_function(self, belief_state:BeliefState, action:Action) -> float:
        if belief_state.velocity < 10 and action.gas:
           return 1
        if action.steer == 0:
            return 1
        return 0
        #raise NotImplementedError
    
RLExperiment(
    environment = Racetrack(),
    agent = Agent(
        sensor = DriverView(),
        cognitive_map = DriverMind(),
        policy = DriverDecisions(action_space_path='racecar_actions.csv'),
    ),
).run(number_of_steps=1000)