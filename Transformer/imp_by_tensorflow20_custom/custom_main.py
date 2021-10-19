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
            inputs = tf.keras.layers.Input((None,)