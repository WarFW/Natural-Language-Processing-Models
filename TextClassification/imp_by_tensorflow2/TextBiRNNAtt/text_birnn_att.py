# -*- coding: utf-8 -*-
# @Time : 2020/4/21 13:50
# @Author : zdqzyx
# @File : text_birnn_att.py
# @Software: PyCharm


import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional
from tensorflow.keras import Model
from TextClassification.imp_by_tensorflow2.TextBiRNNAtt.attention import Attention

def point_wise_feed_forward_network(dense_size):
    ffn = tf.keras.Sequential()
    for size in dense_size:
        ffn.add(Dense(size, activation='relu'))
    return ffn

class TextBiRNNAtt(Model):

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
        super(TextBiRNNAtt, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size

        self.embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(units=128, activation='tanh', return_sequences=True), merge_mode='concat' ) # LSTM or GRU
        self.attention = Attention()
        if self.dense_size 