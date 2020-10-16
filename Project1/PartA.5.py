#
# Project 1, starter code part a
#
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# constants
NUM_CLASSES = 3
# Increase epochs here for better training result and comparison result
epochs = 500
# Change here for testing different batch size
batch_size = 32
num_neurons = 10
decay = 10e-6
seed = 10
test_size = 0.3

np.random.seed(seed)
tf.random.set_seed(seed)

histories = {}
overall_accuracy = {}
#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
trainY = train_Y-1

# Split the data randomly into 7:3 training set and test set
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size = test_size, random_state=1)

# Create the callback function
class MetricCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def create_4layer_model():
    # create the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(num_neurons, input_dim=21, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(decay)))

    # Addming one additional layer here
    model.add(keras.layers.Dense(num_neurons, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(decay)))


    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros'
                                 ))
    optimizer = keras.optimizers.SGD(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


def create_3layer_model():
    # create the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(num_neurons, input_dim=21, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(decay)))

    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros'
                                 ))
    optimizer = keras.optimizers.SGD(learning_rate=0.01)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model


# plot training and validation accuracy for 4-layer model
four_layer_model = create_4layer_model()
histories['4_layer'] = four_layer_model.fit(train_X, train_Y,
                                epochs=epochs,
                                verbose = 2,
                                batch_size=32,
                                validation_data=(test_X,test_Y))


plt.figure()
plt.plot(histories['4_layer'].history['accuracy'], label='training accuracy')
plt.plot(histories['4_layer'].history['val_accuracy'], label='validation accuracy')
plt.ylabel('Overall accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

# train optimal 3-layer model and compare it with 4-layer model
three_layer_model = create_3layer_model()
histories['3_layer'] = three_layer_model.fit(train_X, train_Y,
                                epochs=epochs,
                                verbose = 2,
                                batch_size= 8,
                                validation_data=(test_X,test_Y))

plt.figure()
plt.plot(histories['4_layer'].history['accuracy'], label='4-layer training accuracy')
plt.plot(histories['3_layer'].history['accuracy'], label='3-layer training accuracy')
plt.ylabel('Overall accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

three_evaluate = three_layer_model.evaluate(x=test_X, y=test_Y, verbose=2, return_dict=True)
four_evaluate = four_layer_model.evaluate(x=test_X, y=test_Y, verbose=2, return_dict=True)
print(three_evaluate, four_evaluate)
