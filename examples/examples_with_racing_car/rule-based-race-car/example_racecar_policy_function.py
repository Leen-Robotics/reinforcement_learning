from gym import make
from typing import Optional, Tuple, List
from numpy import asarray
from PIL import Image
from scipy.spatial.distance import cosine
#=======================
from leen_rl import Agent
from leen_rl import RLExperiment
from leen_rl import Environment
from leen_rl import Sensor
from leen_rl import CognitiveMap
from leen_rl import PolicyFunction
from leen_rl import Action
from leen_rl import Observation
from leen_rl import BeliefState
#=======================

class Racetrack(Environment):
    def initialise_state(self) -> None:
        self.environment = make("CarRacing-v0")
        self.pixels = self.environment.reset()
        self.environment.render()

    def update_state(self, action:Action) -> None:
        self.pixels,_, _, _ = self.environment.step(
            action = [action.steer, action.gas, action.brake]
        )
        self.environment.render()

class DriverView(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            pixels = state.pixels,
        )

class DriverMind(CognitiveMap):
    def _reduce_size_of_pixels(
        self,
        pixels:List[List[List[float]]],
        reduced_size:Tuple[int,int],
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
                pixels = self._reduce_size_of_pixels(
                    pixels=pixels, 
                    reduced_size=(8,8)
                )
            )
        )

    def _identify_road_from_pixels(self, pixels:List[float]) -> str:
        pixel_features = {
            "STRAIGHT_ROAD" : [
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
            ],
            "GRADUAL_LEFT" : [
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
            ],
            "GRADUAL_RIGHT" : [
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
            ],
            "SHARP_LEFT" : [
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 0, 0, 1, 1, 1, 1,
            ],
            "SHARP_RIGHT" : [
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 0, 0, 0,
                1, 1, 1, 1, 0, 0, 0, 0,
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
                1, 1, 1, 1, 0, 0, 1, 1,
            ],
        }
        return min(
            pixel_features.keys(),
            key = lambda feature_name: cosine(
                u = pixel_features.get(feature_name),
                v = pixels
            )
        )

    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState]=None,
        previous_action:Optional[Action]=None
    ) -> BeliefState:
        
        reduced_pixels = self._feature_engineer_map(pixels=observation.pixels[:-15,:,:])
        reduced_flattened_pixels = reduced_pixels.flatten()
        return BeliefState(
            previous_turn = previous_action.steer if previous_action else 0,
            reduced_pixels = reduced_pixels,
            current_road_type = self._identify_road_from_pixels(pixels = reduced_flattened_pixels),
            not_moved = previous_belief_state and all(
                reduced_flattened_pixels == previous_belief_state.reduced_pixels.flatten()
            )
        )

class DriverDecisions(PolicyFunction):
    def infer_action(self, belief_state:BeliefState) -> Action:
        LEFT,LEFT_SLOWER,LEFT_FASTER,_,STEADY,SLOWER,FASTER,_,RIGHT,RIGHT_SLOWER,RIGHT_FASTER,_ = self.action_space
        if belief_state.current_road_type == "STRAIGHT_ROAD" and belief_state.not_moved:
            return FASTER
        if belief_state.current_road_type in ("GRADUAL_LEFT", "SHARP_RIGHT"):
            return RIGHT
        if belief_state.current_road_type in ("GRADUAL_RIGHT", "SHARP_LEFT"):
            return LEFT
        return STEADY
        
RLExperiment(
    environment = Racetrack(),
    agent = Agent(
        sensor = DriverView(),
        cognitive_map = DriverMind(),
        policy = DriverDecisions(action_space_path='racecar_actions.csv'),
    ),
).run(number_of_steps=10000)