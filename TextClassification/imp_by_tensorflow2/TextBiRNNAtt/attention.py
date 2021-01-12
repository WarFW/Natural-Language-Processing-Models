# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:55
# @Author : zdqzyx
# @File : attention.py
# @Software: PyCharm

from tensorflow.keras import  initializers,regularizers,constraints
from  tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs
                 ):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
      