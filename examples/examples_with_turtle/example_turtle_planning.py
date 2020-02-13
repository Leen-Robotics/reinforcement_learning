from turtle import Screen, Turtle
from copy import deepcopy
from math import cos, sin, radians
from scipy.spatial.distance import euclidean
from typing import Optional, List
#=======================
from leen_rl import Agent
from leen_rl import RLExperiment
from leen_rl import Environment
from leen_rl import Sensor
from leen_rl import CognitiveMap
from leen_rl import GoalPlanning
from leen_rl import Action
from leen_rl import Observation
from leen_rl import BeliefState
#=======================

class TurtleWorld(Environment):
    def initialise_state(self) -> None:
        screen = Screen()
        self.target = Turtle()
        self.target.shape("circle")
        self.target.penup()
        self.target.setpos(150,-50)
        self.target.color("red")
        self.agent = Turtle()
        self.agent.shape("turtle")
        self.agent_speed = 0
        self.step_size = 10

    def update_state(self, action:Action) -> None:
        self.agent_speed += action.acceleration
        self.agent.right(action.turn*self.step_size)
        self.agent.fd(self.agent_speed)
            
class TurtleSense(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            position = state.agent.position(), 
            heading = state.agent.heading(),
            velocity = state.agent_speed,
        )

class TurtleMind(CognitiveMap):
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState] = None,
        previous_action:Optional[Action] = None
    ) -> BeliefState:

        return BeliefState(
            acceleration = previous_action.acceleration if previous_action else 0,
            turn = previous_action.turn if previous_action else 0,
            previous_velocity = previous_belief_state.velocity if previous_belief_state else 0,
            previous_heading = previous_belief_state.heading if previous_belief_state else 0,
            previous_position = previous_belief_state.position if previous_belief_state else 0,
            velocity = observation.velocity,
            heading = observation.heading,
            position = observation.position,
        )

class TurtleAct(GoalPlanning):
    target_position = (150,-50)

    def _distance(self, vector1:List[float], vector2:List[float]) -> float:
        return 1 - euclidean(vector1, vector2)

    def reward_function(self, belief_state:BeliefState) -> float:
        reward = self._distance(
            vector1 = belief_state.position, 
            vector2 = self.target_position
        )
        return reward

    def state_transition_function(self, action:Action, belief_state:BeliefState) -> BeliefState:
        x,y = belief_state.position
        theta = radians(belief_state.heading) 
        x_next = x + (cos(theta)*belief_state.velocity)
        y_next = y + (sin(theta)*belief_state.velocity)

        next_belief_state = deepcopy(belief_state)
        next_belief_state.acceleration = action.acceleration 
        next_belief_state.turn = action.turn 
        next_belief_state.previous_velocity = belief_state.velocity
        next_belief_state.previous_heading = belief_state.heading 
        next_belief_state.previous_position = belief_state.position
        next_belief_state.velocity = belief_state.velocity + action.acceleration
        next_belief_state.heading = belief_state.heading + action.turn
        next_belief_state.position = (x_next, y_next)
        return next_belief_state 

RLExperiment(
    environment = TurtleWorld(),
    agent = Agent(
        sensor = TurtleSense(),
        cognitive_map = TurtleMind(),
        policy = TurtleAct(action_space_path='turtle_actions.csv', max_depth=15, sample_size=1),
    ),
).run(number_of_steps=400)