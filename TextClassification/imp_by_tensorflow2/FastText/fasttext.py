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
                    last_activation = 'softmax'
                ):
        super(FastText, self).__init__()
        self.maxlen = maxlen
        # self.max_features = max_features
        # self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(128, activation='relu')
        self.clas