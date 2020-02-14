# Leen RL
A simple interface for writing modular RL experiments

![alt text](leen_rl/architecture.png)
Everything in red must is defined by the user (depending on the nature of the experiment).  The rest is handled in the background. 

## RLExperiment
Here is an example walk-through of how to set-up a leen rl experiment
```python
from leen_rl import RLExperiment

your_experiment = RLExperiment(
    environment = your_environment,
    agent = your_agent,
)
your_experiment.run(number_of_steps=1000)
```
TODO: functionality for learning and plotting

### Environment
When setting up your environment, you need only define two functions: `initialise_state()` and `update_state()` which is where you can define or integrate any environment into your experiment 
```python
from leen_rl import Environment

class YourWorld(Environment):
  def initialise_state(self) -> None:
    pass
    
  def update_state(self, action:Action) -> None:
    pass

your_environment = YourWorld()
```
We provide a few examples for you, some of which use environments from openAI gym or turtles, etc
![alt text](https://raw.githubusercontent.com/leen-robotics/reinforcement_learning/master/examples/examples_with_racing_car/racecar.gif)
```python
from gym import make

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

your_environment = RaceTrack()
```

![alt text](https://raw.githubusercontent.com/leen-robotics/reinforcement_learning/master/examples/examples_with_turtle/turtle_demo.gif)
```python
from turtle import Screen, Turtle

class TurtleWorld(Environment):
    def initialise_state(self) -> None:
        screen = Screen()
        self.agent = Turtle()
        self.agent_speed = 0
        self.step_size = 10

    def update_state(self, action:Action) -> None:
        self.agent_speed += action.acceleration
        self.agent.right(action.turn*self.step_size)
        self.agent.fd(self.agent_speed)

your_environment = TurtleWorld()
```
If you wish you may define a custom environment for your experiment

### Agent
The `agent` is comprised of three parts: the `sensor`, `cognitive_map` and `policy`  

```python
from leen_rl import Agent

your_agent = Agent(
   sensor = your_sensor,
   cognitive_map = your_cognitive_map,
   policy = your_policy,
)
```

#### Sensor
The `sensor` maps the environment's state to an observation.  It is where you specify which signals (from the environment) the agent has access to (it may only able to see partial information from the environment's state, etc). 

```python
from leen_rl import Sensor, Environment, Observation

class YourAgentsSensor(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        pass
 
your_sensor = YourAgentsSensor()
```

Since this is how the agent will get information from the environment and is thus dependent on the environment. For example, in the racecar example above, the agent has access to the environment's pixels (note: the `Observation` class can take in any keyword arguments you like, e.g. `pixels`)
```python
class DriverView(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            pixels = state.pixels
        )

your_sensor = DriverView()
```
For the turtle example, the sensor allows the agent to observe its position, heading and velocity in the environment.
```python
class TurtleSense(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            position = state.agent.position(), 
            heading = state.agent.heading(),
            velocity = state.agent_speed,
        )
your_sensor = TurtleSense()
```
The amount of information which the agent has access to is left for you to define.  (Note: no feature engineering should be done to the environmental state information at this stage)

#### CognitiveMap
The `cognitive map` maps the agent's raw observation to an internal representation (known as the belief state) which can consist of a completely abstract space or engineered features, etc. You may wish to combine past belief states for contextual information too.  

```python
from leen_rl import CognitiveMap, BeliefState, Observation

class YourAgentsMind(CognitiveMap):
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState] = None,
        previous_action:Optional[Action] = None
    ) -> BeliefState:
        pass

your_cognitive_map = YourAgentsMind()
```

Or you may wish to skip this step altogether, in which case you can pass out a belief state with exactly the same information as the observation.
```python
class SkipThisPart(CognitiveMap):
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState] = None,
        previous_action:Optional[Action] = None
    ) -> BeliefState:
        return BeliefState(**observation.__dict__)

your_cognitive_map = SkipThisPart()
```
 
#### Policy
The `policy` maps the belief states to an action using whichever method you define.
```python
from leen_rl import Policy, BeliefState, Action

class YourAgentsPolicy(Policy):
    def get_action(self, belief_state:BeliefState) -> Action:
        raise NotImplementedError

your_policy = YourAgentsPolicy(action_space_path = 'path/to/your/actions.csv`)
```

Aswell as defining a custom policy, there are various kinds of pre-defined Policies to choose from, including `PolicyFunction`, `ValueFunction`, `QFunction`, `GoalPlanning`, etc.
For these you won't need to define the `get_action` function but you will need to define one or two other functions, such as the `q_function` for `QFunction`:
```python
from leen_rl import QFunction

class YourAgentsPolicy(QFunction):
    def q_function(self, belief_state:BeliefState, action:Action) -> float:
        pass

your_policy = YourAgentsPolicy(action_space_path = 'path/to/your/actions.csv`)
```
or the `state_transition_function` for the `ValueFunction` or `GoalPlanning` policies, as well as the `value_function` or `reward_function` respectively:
```python
from leen_rl import ValueFunction

class YourAgentsPolicy(ValueFunction):
    def state_transition_function(self, action:Action, belief_state:BeliefState) -> BeliefState:
        pass

    def value_function(self, belief_state:BeliefState) -> float:
        pass

your_policy = YourAgentsPolicy(action_space_path = 'path/to/your/actions.csv`)
```
or alternatively try the random policy as a baseline (which needs nothing defining at all)
```python
from leen_rl import RandomExploration

your_policy = RandomExploration(action_space_path = 'path/to/your/actions.csv`)
```

You will also have noticed that a path to the action space needs to be set for any policy.  This is simply a csv file of the possible actions the agent can do.  This will create a set of `Action` objects with the attributes and values specified in the file.  e.g.
```csv
steer,gas,brake
-1,0,0
-1,1,1
0,0,0
0,0,1
```
will produce four `Action` type actions with the attributes `steer`, `gas` and `brake`

TODO: allow for continual action spaces to be defined
TODO: integrate intrinsic rewards
TODO: PID controller in predefined policies
