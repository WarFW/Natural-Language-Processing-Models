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
from TextClassification.imp_by_tensorflow2.TextBiRNNAtt.text_birnn_att import TextBiRNNAtt

def checkout_dir(dir_path, 