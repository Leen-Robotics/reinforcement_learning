from pandas import DataFrame
from numpy.linalg import pinv
from typing import Dict, Tuple
from random import choice
#=======================
from leen_rl import PolicyFunction, Action, BeliefState
#=======================

class MatrixFactorisationClassifier(PolicyFunction):
    def learn(self, input_vectors_train:DataFrame, output_vectors_train:DataFrame) -> None:
        self.learnt_output_vectors = fit_output_embeddings_to_input_embedding_space(
            coocurrence_matrix = output_vectors_train,
            input_embedding_matrix = input_vectors_train
        )

    def infer_action(self, belief_state:BeliefState) -> Action:
        WARMUP_DELAY = 1

        _,samples = belief_state.past_input_vectors.shape
        if samples>WARMUP_DELAY:
            
            if belief_state.updated:
                self.learn(
                    input_vectors_train = belief_state.past_input_vectors,
                    output_vectors_train = belief_state.past_output_vectors
                )

            labels = self._infer_label(
                input_vectors_test = belief_state.new_batch_of_input_vectors_to_classify,
            )
            if labels: 
                predicted_label = choose_a_label(labels)     
                return max(
                    self.action_space,
                    key = lambda action : action.label == predicted_label
                )
        return choice(self.action_space)

    def _infer_label(self, input_vectors_test:DataFrame) -> DataFrame:
        return find_most_similar_output_to_given_input_by_looking_up_coocurrence_matrix(
            coocurrence_matrix = produce_cooccurrence_matrix_of_inputs_and_outputs_in_shared_embedding_space_using_dot_product(
                input_vectors_in_shared_embedding_space = input_vectors_test,
                output_vectors_in_shared_embedding_space = self.learnt_output_vectors,
            )
        )        
    
def fit_output_embeddings_to_input_embedding_space(
    coocurrence_matrix:DataFrame,
    input_embedding_matrix:DataFrame,
    ) -> DataFrame:

    pseudo_inverse_of_input_matrix = pinv(input_embedding_matrix.T)  
    output_embedding_matrix = DataFrame(
        pseudo_inverse_of_input_matrix @ coocurrence_matrix
    )
    output_embedding_matrix.columns = coocurrence_matrix.columns
    return output_embedding_matrix

def produce_cooccurrence_matrix_of_inputs_and_outputs_in_shared_embedding_space_using_dot_product(
    input_vectors_in_shared_embedding_space:DataFrame,
    output_vectors_in_shared_embedding_space:DataFrame,
    ) -> DataFrame:

    return normalise_matrix(
        matrix=input_vectors_in_shared_embedding_space.T @ output_vectors_in_shared_embedding_space
    ) 
        
def normalise_matrix(matrix:DataFrame) -> DataFrame:
    return matrix.div(matrix.sum(axis=1), axis=0)

def find_most_similar_output_to_given_input_by_looking_up_coocurrence_matrix(
    coocurrence_matrix:DataFrame
) -> Dict[str,Tuple[str,float]]:

    return {
        sentence: [
            (label,confidence) for label,confidence in coocurrence_matrix.T[sentence].sort_values(
                ascending=False
            ).items() if confidence > 0
        ] for sentence in coocurrence_matrix.index
    }

def choose_a_label(labels:Dict[str,Tuple[str,float]]) -> str:
    actions_and_confidences = [actions[0] for actions in labels.values()]
    action,_  = actions_and_confidences[0]
    return action
