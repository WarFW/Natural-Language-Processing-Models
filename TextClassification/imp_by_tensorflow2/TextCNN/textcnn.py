# -*- coding: utf-8 -*-
# @Time : 2020/4/20 14:44
# @Author : zdqzyx
# @File : textcnn.py
# @Software: PyCharm

import  tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.ker