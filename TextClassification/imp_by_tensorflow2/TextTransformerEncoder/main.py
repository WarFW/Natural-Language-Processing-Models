# -*- coding: utf-8 -*-
# @Time : 2020/4/21 11:44
# @Author : zdqzyx
# @File : main.py
# @Software: PyCharm


# ===================== set random  ===========================
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(0)
rn.seed(0)
tf.random.set_seed(0)

# =============================================================

import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from TextClassification.imp_by_tensorflow2.TextTransformerEncoder.modeling import TextTransformerEncoder

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)


class ModelHepler:
    def __init__(self, class_num, maxlen, max_features, embedding_dims, epochs, batch_size, num_layers,
                 num_heads,
                 dff,
                 pe_input,):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.pe_input = pe_input

    def create_model(self):
        input_seq = tf.keras.layers.Input((self.maxlen, ), dtype=tf.int64, name='inputs_seq')
        input_mask = tf.keras.layers.Input((1,1,self.maxlen), dtype=tf.float32, name='inputs_mask')
        internal_model = TextTransformerEncoder(maxlen=self.maxlen,
                         max_features=self.max_features,
        