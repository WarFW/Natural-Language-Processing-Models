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
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def scaled_dot_product_attention(q, k, v, mask=None):
    '''计算attention
    q,k,v的第一维度必须相同
    q,k的最后一维必须相同
    k,v在倒数第二的维度需要相同, seq_len_k = seq_len_q=seq_len。
    参数:
    q: 请求的形状 == (..., seq_len_q, d)
    k: 主键的形状 == (..., seq_len, d)
    v: 数值的形状 == (..., seq_len, d_v)
    mask: Float 张量，其形状能转换成
       