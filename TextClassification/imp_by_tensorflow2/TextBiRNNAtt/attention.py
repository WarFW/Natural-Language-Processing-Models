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
    