# Leen RL
simple interface for rl experiments

Everything in red must be defined by the user depending on the use case.  The rest is all handled in the background. 
- ![alt text](leen_rl/architecture.png)

Here is an example walk-through of how to set-up a leen rl experiment

## RLExperiment

TODO: functionality for learning and plotting

## Environment
integrate any environment into your experiment. 

- e.g. Open AI Gym
- ![alt text](https://raw.githubusercontent.com/leen-robotics/reinforcement_learning/master/examples/examples_with_racing_car/racecar.gif)
- e.g. Turtles
- ![alt text](https://raw.githubusercontent.com/leen-robotics/reinforcement_learning/master/examples/examples_with_turtle/turtle_demo.gif)

## Agent

TODO: integrate intrinsic rewards

## Sensor

## CognitiveMap

## Policy
Aswell as defining a custom policy, there are various kinds of pre-defined Policies to choose from, including PolicyFunction, ValueFunction, QFunction, GoalPlanning, etc.

TODO: PID controller in predefined policies

Actionspaces
TODO: allow for continual action spaces to be defined
