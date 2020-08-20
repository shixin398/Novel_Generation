# -*- coding: utf-8 -*-

"""
Created on 2020-08-19 16:43
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import numpy as np
from utils import CorpusReader
import sys

class BidirectionalGRU(object):

    def __init__(self, all_words, token_dict, max_length):
        self.token_dict = token_dict
        self.vocab_size = len(token_dict)
        self.max_length = max_length
        self.all_words = all_words

    def model(self):
        model_input = layers.Input(shape=(self.max_length, ), dtype='float32')
        embedding_layer = layers.Embedding(self.vocab_size, 128, input_length=(self.max_length))(model_input)
        biGRU1 = layers.Bidirectional(layers.GRU(256, return_sequences=True))(embedding_layer)
        biGRU2 = layers.Bidirectional(layers.GRU(128))(biGRU1)
        model_output = layers.Dense(self.vocab_size, activation='softmax')(biGRU2)
        model = keras.models.Model(model_input, model_output)

        optimizer = optimizers.RMSprop(lr=3e-3)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        model.summary()
        return model


