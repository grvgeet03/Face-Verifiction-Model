#Custom L1 Distance Layer Module

import tensorflow as tf
from tensorflow.keras.layers import Layer # type: ignore


# Custom L1 Distance Layer from Jupyter Notebook
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        # Ensure that you're working with the first element if they are lists
        if isinstance(input_embedding, list):
            input_embedding = input_embedding[0]
        if isinstance(validation_embedding, list):
            validation_embedding = validation_embedding[0]
        return tf.math.abs(input_embedding - validation_embedding)
    