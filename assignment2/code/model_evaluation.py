import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models

algos = ['SGD-momentum', 'RMSProp', 'Adam', 'dropouts']

# Fixed, no need change
def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32) / 255
    labels = np.array(labels, dtype=np.int32)
    return data, labels

batch_size = 128
num_ch_c1 = [10, 30, 50, 70, 90]  # Question 2
num_ch_c2 = [20, 40, 60, 80, 100]  # Question 2
optimizer_ = 'SGD'  # Question 3

x_train, y_train = load_data('data_batch_1')
x_test, y_test = load_data('test_batch_trim')


for c1 in num_ch_c1:
    for c2 in num_ch_c2:
        model = keras.models.load_model(f'.\models\{c1}_{c2}_{optimizer_}_no_dropout')
        score = model.evaluate(x = x_test,y = y_test, verbose=0, batch_size = batch_size)
        print(f'Model {c1}_{c2}_{optimizer_}_no_dropout score: {score}')


for optimizer in algos:
    if optimizer == 'dropouts':
        model = keras.models.load_model(f'.\models\\70_20_{optimizer}_dropout')
    else:
        model = keras.models.load_model(f'.\models\\70_20_{optimizer}_no_dropout')
    score = model.evaluate(x=x_test, y=y_test, verbose=0, batch_size=batch_size)
    print(f'Model 70_20_{optimizer}_no_dropout score: {score}')