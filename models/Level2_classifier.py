# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 03:30:40 2022

@author: harsh
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import tensorflow
import keras

from util.dataset_generator import *
from util.metrics import *
import src.global_variables as GV

from pyparsing import actions
# import
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Dropout,
    LSTM,
    GRU,
    Bidirectional
)

# GPU optimization
tensorflow.config.run_functions_eagerly(True)
tensorflow.data.experimental.enable_debug_mode()


from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import Input

class ActionClassifier2:

    @staticmethod
    def build(timesteps=15, dim=8576, classes=17):
        model = Sequential()
        model.add(Bidirectional(GRU(32, return_sequences=True),
                                input_shape=(timesteps, dim),
                                merge_mode='sum'))
        # model.add(Dropout(0.3))
        model.add(Bidirectional(GRU(32), merge_mode='sum'))
        model.add(Flatten())

        # Add a Dense layer.
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(classes, activation="softmax"))

        return model