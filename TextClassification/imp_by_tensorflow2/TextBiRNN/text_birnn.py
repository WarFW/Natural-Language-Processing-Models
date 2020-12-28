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
    ffn = tf.k