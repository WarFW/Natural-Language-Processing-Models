# -*- coding: utf-8 -*-
# @Time : 2020/4/21 10:47
# @Author : zdqzyx
# @File : text_birnn.py
# @Software: PyCharm

import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras import backend as K


def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn


class TextBiRNN(Model):

    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
     