from typing import Optional, Dict, List, Tuple
from pandas import DataFrame, Index
from PIL.Image import fromarray
from numpy import asarray
#=======================
from leen_rl import CognitiveMap, Action, Observation, BeliefState
#=======================
class DriverMind(CognitiveMap):
    
    def get_belief_state(
        self, 
        observation:Observation,
        previous_belief_state:Optional[BeliefState]=None,
        previous_action:Optional[Action]=None
    ) -> BeliefState:

        SLIDING_WINDOW_TIME_DELAY = 100

        new_data_sample = observation.reward_signal > 0
        if new_data_sample or not previous_belief_state:
            output_vectors_over_time = get_past_action_vectors(
                score = observation.reward_signal,
                action = previous_action,
                belief_state = previous_belief_state
            )[:SLIDING_WINDOW_TIME_DELAY]
            input_vectors_over_time = get_past_state_vectors(
                output_vectors = output_vectors_over_time,
                belief_state = previous_belief_state,
                sample_size = SLIDING_WINDOW_TIME_DELAY,
            )
        else:
            output_vectors_over_time = previous_belief_state.past_output_vectors
            input_vectors_over_time = previous_belief_state.past_input_vectors

        return BeliefState(
            updated = new_data_sample,
            past_input_vectors = input_vectors_over_time,
            past_output_vectors = output_vectors_over_time,
            new_batch_of_input_vectors_to_classify = embed_batch_of_pixels(
                batch_of_pixels=[observation.pixels]
            )
        )

def get_past_action_vectors(
    score:float,
    belief_state:Optional[BeliefState],
    action:Optional[Action],
) -> DataFrame:

    if belief_state and action: 
        return belief_state.past_output_vectors.append(
            [
                one_hot_encode_action(
                    action=action,
                    weight = score
                )
            ]
        )    
    return DataFrame(columns = ["LEFT","RIGHT","FASTER","SLOWER"])

def one_hot_encode_action(action:Action, weight:float=1.) -> Dict[str,float]:
    return {
        "LEFT":float(action.steer < 0) * weight,
        "RIGHT":float(action.steer > 0) * weight,
        "FASTER":float(action.gas > 0) * weight,
        "SLOWER":float(action.brake > 0) * weight
    }

def get_past_state_vectors(
    output_vectors:Optional[DataFrame],
    belief_state:Optional[BeliefState],
    sample_size:int,
) -> DataFrame:
    if belief_state:
        return belief_state.past_input_vectors.T.append(
            belief_state.new_batch_of_input_vectors_to_classify[0]
        )[-sample_size:].T 
    return DataFrame(columns = output_vectors.index)


def embed_batch_of_pixels(batch_of_pixels:List[List[List[List[float]]]]) -> DataFrame:
    return DataFrame(
        {
            sample_number:embed_pixels(pixels=pixels) for sample_number,pixels in enumerate(batch_of_pixels)
        }
    ) 

def embed_pixels(pixels:List[List[List[float]]]) -> List[float]:
    return mini_greyscale_version(
        pixels=pixels,
        reduced_size = (8,8)
    ).flatten()

def mini_greyscale_version(
    pixels:List[List[List[float]]], 
    reduced_size:Tuple[int,int],
) -> List[List[float]]:

    return normalise_pixels(
        pixels = flatten_rgb_channels(
            pixels = reduce_size_of_pixels(
                pixels=pixels, 
                reduced_size=reduced_size
            )
        )
    )

def normalise_pixels(pixels:List[List[float]]) -> List[List[float]]:
    return pixels * 1/255

def flatten_rgb_channels(
    pixels:List[List[List[float]]],
    rgb_channel_weights:Tuple[float,float,float] = (.2989,.587,.114),
) -> List[List[float]]:

    return sum(
        rgb_channel_weights[index]*pixels[:,:,index] for index in range(3)
    )

def reduce_size_of_pixels(
    pixels:List[List[List[float]]],
    reduced_size:Tuple[int,int],
) -> List[List[List[float]]]:

    return asarray(
        fromarray(pixels).resize(reduced_size)
    )