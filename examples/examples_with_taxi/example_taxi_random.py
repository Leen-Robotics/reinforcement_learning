from gym import make
from typing import Optional, Tuple
#=======================
from leen_rl import Agent
from leen_rl import RLExperiment
from leen_rl import Environment
from leen_rl import Sensor
from leen_rl import CognitiveMap
from leen_rl import RandomExploration
from leen_rl import Action
from leen_rl import Observation
from leen_rl import BeliefState
#=======================

class TaxiWorld(Environment):
    def initialise_state(self) -> None:
        self.environment = make("Taxi-v3")
        encoded_state = self.environment.reset()
        self.signals = self.environment.decode(encoded_state)
        self.reward = 0
        self.environment.render()

    def update_state(self, action:Action) -> None:
        encoded_state,reward,_,_ = self.environment.step(action = action.index)
        self.signals = self.environment.decode(encoded_state)
        self.reward = reward
        self.environment.render()


class TaxiSense(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            raw_signals = list(state.signals),
            extrinsic_reward = state.reward
        )

class TaxiMind(CognitiveMap):
    
    def _feature_engineer_index_to_coordinates(
        self, 
        index:int,
        index_coordinate_mapping:Tuple[Tuple[int,int]] = (
            (0,0),(0,4),
            (4,0),(4,3)
        )
    ) -> Tuple[int,int]:
        return index_coordinate_mapping[index]

    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState]=None,
        previous_action:Optional[Action]=None
    ) -> BeliefState:

        taxi_x, taxi_y, passenger_index, destination_index = observation.raw_signals
        taxi_coordinates = (taxi_x, taxi_y)
        passenger_coordinates = taxi_coordinates if passenger_index == 4 else self._feature_engineer_index_to_coordinates(
            index=passenger_index
        )
        destination_coordinates = taxi_coordinates if destination_index == 4 else self._feature_engineer_index_to_coordinates(
            index=destination_index
        )
        return BeliefState(
            destination_location = destination_coordinates,
            passenger_location = passenger_coordinates,
            taxi_location = taxi_coordinates,
        )

RLExperiment(
    environment = TaxiWorld(),
    agent = Agent(
        sensor = TaxiSense(),
        cognitive_map = TaxiMind(),
        policy = RandomExploration(action_space_path='taxi_actions.csv'),
    ),
).run(number_of_steps=10)