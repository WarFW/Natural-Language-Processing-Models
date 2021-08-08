# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:50
# @Author : zdqzyx
# @File : text_selfatt.py
# @Software: PyCharm


import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, Flatten
from tensorflow.keras import Model
from TextClassification.imp_by_tensorflow2.TextSelfAtt.attention import MultiHeadAttention

def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn

class TextSelfAtt(Model):

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
        :param class_num:
        :param last_activation:
        '''
        super(TextSelfAtt, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
  