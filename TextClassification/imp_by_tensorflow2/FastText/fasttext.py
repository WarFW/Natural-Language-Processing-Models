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
        self.classifier = Dense(self.class_num, activation=last_activation)

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of FastText must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of FastText must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        pool = self.pooling(emb)
        h = self.dense(pool)
        output = self.classifier(h)
        return output

    def build_graph(self, input_shape):
        i