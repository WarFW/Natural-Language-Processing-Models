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
                 max_featur