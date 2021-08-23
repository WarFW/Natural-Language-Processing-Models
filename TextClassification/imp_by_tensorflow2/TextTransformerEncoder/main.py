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
from tens