# -*- coding: utf-8 -*-
# @Time : 2020/4/22 21:46
# @Author : zdqzyx
# @File : main.py
# @Software: PyCharm

import tensorflow_datasets as tfds
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from Transformer.imp_by_tensorflow20_custom.modeling import Transformer
import os

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_exam