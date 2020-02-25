from gym import make
#=======================
from leen_rl import Environment, Sensor, Action, Observation
#=======================
class Racetrack(Environment):
    def initialise_state(self) -> None:
        self.environment = make("CarRacing-v0")
        self.pixels = self.environment.reset()
        self.environment.render()
        self.reward = 0.

    def update_state(self, action:Action) -> None:
        self.pixels,self.reward, _, _ = self.environment.step(
            action = [action.steer, action.gas, action.brake]
        )
        self.environment.render()

class DriverView(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            pixels = state.pixels,
            reward_signal = state.reward,
        )