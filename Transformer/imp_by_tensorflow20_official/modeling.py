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
          (..., seq_len_q, seq_len)。默认为None。
    返回值:
    输出，注意力权重
    '''
    # (batch_size, num_heads, seq_len_q, d ) dot (batch_size, num_heads, d, seq_ken_k) = (batch_size, num_heads,, seq_len_q, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 缩放matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    # 将 mask 加入到缩放的张量上。
    if mask is not None:
        # (batch_size, num_heads,, seq_len_q, seq_len) + (batch_size, 1,, 1, seq_len)
        scaled_attention_logits += (mask * -1e9)

    # softmax归一化权重 (batch_size, num_heads, seq_len)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # seq_len_q个位置分别对应v上的加权求和
    # (batch_size, num_heads, seq_len) dot (batch_size, num_heads, d_v) = (batch_size, num_heads, seq_len_q, d_v)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert (d_model > num_heads) and (d_model % num_heads == 0)
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

    