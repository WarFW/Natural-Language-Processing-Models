# -*- coding: utf-8 -*-
# @Time : 2020/4/22 16:28
# @Author : zdqzyx
# @File : modeling.py
# @Software: PyCharm

import tensorflow as tf
import  numpy as np

def get_angles(pos, i, d_model):
    '''
    :param pos:单词在句子的位置
    :param i:单词在词表里的位置
    :param d_model:词向量维度大小
    :return:
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    '''
    :param position: 最大的position
    :param d_model: 词向量维度大小
    :return: [1, 最大position个数，词向量维度大小] 最后和embedding矩阵相加
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
    