# -*- coding: utf-8 -*-
# @Time : 2020/4/24 21:07
# @Author : zdqzyx
# @File : custom_main.py
# @Software: PyCharm

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from Transformer.imp_by_tensorflow20_custom.modeling import Transformer

def create_model(params, is_train):
    """Creates transformer model."""
    with tf.name_scope("model"):
        if is_train:
            inputs = tf.keras.layers.Input((None,), dtype='int64', name="inputs")
            targets = tf.keras.layers.Input((None,), dtype='int64', name="targets")
            internal_model = Transformer(params,)
            logits = internal_model([inputs, targets], training=is_train)
            # logits = tf.keras.layers.Lambda(lambda x: x, name="logits", dtype=tf.float32)(logits)
            model = tf.keras.Model([inputs, targets], logits)
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
            )
            # model.build_