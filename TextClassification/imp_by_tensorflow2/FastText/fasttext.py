# -*- coding: utf-8 -*-
# @Time : 2020/4/20 14:16
# @Author : zdqzyx
# @File : fasttext.py
# @Software: PyCharm


import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model

class FastText(Model):

    def __init__(self,
                    maxlen,
                    max_features,
                    embedding_dims,
                    class_num,
                    last_activation = 's