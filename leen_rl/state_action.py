from json import dumps
from typing import List, Any
from pandas import read_csv
import numpy
#=======================
#=======================
class BeliefState:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = kwargs

    def __str__(self) -> str:
        return f"""{self.__class__.__name__} = {dumps(
            self.__dict__,
            indent=4,
            default= convert_datatype
        )}"""

class Observation:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__name__} = {self.__dict__}"

class Action:
    def __init__(self, **kwargs) -> None:
        self.__dict__ = kwargs

    def __str__(self) -> str:
        return f"{self.__class__.__name__} = {self.__dict__}"

def load_in_action_space(state_space_address:str) -> List[Action]:
    return [
        Action(
            **kwargs
        ) for kwargs in read_csv(state_space_address).to_dict(
            orient='records'
        )
    ]

def convert_datatype(value:Any) -> Any:
    if _is_numpy_datatype(value):
        if _is_numpy_array(value):
            try:
                print(
                    _convert_image_to_ascii(value)
                )
                return "<see printed map>"
            except:
                return list(value)
        return value.item() 
    return str(value)

def _is_numpy_datatype(value:Any) -> bool:
    return type(value).__module__ == numpy.__name__ 

def _is_numpy_array(value:Any) -> bool:
    return type(value).__name__ == "ndarray"

def _convert_image_to_ascii(
    pixels:List[List[float]],
    ascii_scale:str = "@%#*+=-:. ",
    column_separator:str = " ",
    row_separator:str = "\n"
) -> str:    
    max_index = len(ascii_scale)
    return row_separator.join(
        column_separator.join(
            ASCII_scale[int(pixel*max_index)] for pixel in row
        ) for row in pixels
    ) 