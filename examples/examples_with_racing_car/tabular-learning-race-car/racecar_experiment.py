#=======================
from leen_rl import Agent, RLExperiment
from racecar_environment import Racetrack, DriverView
from racecar_belief_state import DriverMind
from matrix_factorisation import MatrixFactorisationClassifier
#=======================
RLExperiment(
    environment = Racetrack(),
    agent = Agent(
        sensor = DriverView(),
        cognitive_map = DriverMind(),
        policy = MatrixFactorisationClassifier(action_space_path='racecar_actions.csv'),
    ),
).run(number_of_steps=10000)