from typing import Optional
from random import choice
from time import sleep
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

class ChatRoom(Environment):
    def _get_reply_template(self, action:Action) -> str:
        self.ready_to_reply = action.intent != "CONFUSED"
        return choice(
            [
                action.template1, 
                action.template2, 
                action.template3
            ]
        )
    def _display_to_user(self, message:str) -> None:
        if self.ready_to_reply:
            print(
                self.chat_window.format(
                    user = self.user_utterance,
                    system = message
                )
            )

    def _fill_buffer_if_empty(self) -> None:
        while not self.user_utterance_buffer:
            self.user_utterance_buffer = input(">").split()

    def _get_next_word_from_buffer(self) -> None:
        sleep(1)
        self.user_utterance = self.user_utterance_buffer.pop(0)

    def initialise_state(self) -> None:
        self.ready_to_reply = False
        self.user_utterance_buffer = []
        self.user_utterance = ""
        self.chat_window = """

            USER: {user}

            SYSTEM: {system}
            
        """

    def update_state(self, action:Action) -> None:
        system_reply = self._get_reply_template(action=action)
        self._display_to_user(message = system_reply)
        self._fill_buffer_if_empty()
        self._get_next_word_from_buffer()
        
class AgentEars(Sensor):
    def get_observation(self, state:Environment) -> Observation:
        return Observation(
            user_utterance = state.user_utterance,
        )

class AgentMind(CognitiveMap):
    INTENTS = {
        "hello":"START",
        "bye":"END",
        "no":"DENY",
        "yes":"AFFIRM",
    }
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState] = None,
        previous_action:Optional[Action] = None
    ) -> BeliefState:

        NEW_UTTERANCE = [observation.user_utterance]
        intent = self.INTENTS.get(observation.user_utterance)
        return BeliefState(
            memory_of_user_utterances = previous_belief_state.memory_of_user_utterances + NEW_UTTERANCE if previous_belief_state else NEW_UTTERANCE,
            previous_intent_user = previous_belief_state.intent_user if previous_belief_state else "",
            intent_system = previous_action.intent if previous_action else "",
            intent_user = intent if intent else "OUT_OF_SCOPE"
        )

class AgentDecisions(PolicyFunction):
    def infer_action(self, belief_state:BeliefState) -> Action:
        BYE,GREET,ASK_NAME,CONFUSED = self.action_space
        if belief_state.intent_user == "START":
            return GREET 
        if belief_state.intent_user == "END":
            return BYE 
        if belief_state.intent_user == "OUT_OF_SCOPE":
            return CONFUSED
        return ASK_NAME


RLExperiment(
    environment = ChatRoom(),
    agent = Agent(
        sensor = AgentEars(),
        cognitive_map = AgentMind(),
        policy = AgentDecisions(action_space_path='chatbot_actions.csv'),
    ),
).run(number_of_steps=10)