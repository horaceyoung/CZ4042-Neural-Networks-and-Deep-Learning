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

epochs = 500
# Change here for testing different batch size
batch_size = 8
num_neurons = 25
decay = [0,1e-3,1e-6,1e-9,1e-12]
seed = 10
test_size = 0.3

np.random.seed(seed)
tf.random.set_seed(seed)

histories = {}
overall_accuracy = {}
#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainY = train_Y-1

# Split the data randomly into 7:3 training set and test set and scale the input data
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size = test_size, random_state=1)
train_X = scale(train_X, np.min(train_X, axis=0), np.max(train_X, axis=0))
test_X = scale(test_X, np.min(test_X, axis=0), np.max(test_X, axis=0))

# Create the callback function
class MetricCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def create_model(decay_parameter):
    # create the model
    model = keras.Sequential()
    model.add(keras.layers.Dense(num_neurons, input_dim=21, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(decay_parameter)))
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

for decay_parameter in decay:
    # Split the data
    n_split = 5
    split_counter = 1
    batch_accuracy = []
    current_time = time.time()
    for train_index, test_index in KFold(n_split).split(train_X):
        model = create_model(decay_parameter)
        train_x, test_x = train_X[train_index], train_X[test_index]
        train_y, test_y = train_Y[train_index], train_Y[test_index]

        histories['fold_'+str(split_counter)] = model.fit(train_x, train_y,
                                                 epochs=epochs,
                                                 verbose = 2,
                                                 batch_size=batch_size)
        split_counter += 1

    # Record the total training time for each batch size
    endtime = time.time() - current_time

    # plot learning accuracy curves for each fold
    plt.figure()
    for i in range(1, 6):
        plt.plot(histories['fold_' + str(i)].history['accuracy'], label='training accuracy fold ' + str(i))
    plt.ylabel('Train accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower right")
    plt.show()

    # plot learning accuracy curve for the mean of all five folds
    plt.figure()
    for i in range(1, 6):
        batch_accuracy.append(histories['fold_' + str(i)].history['accuracy'])
    batch_accuracy = np.mean(batch_accuracy, axis=0)
    overall_accuracy[decay_parameter] = batch_accuracy
    plt.plot(batch_accuracy, label='training accuracy with decay parameter ' + str(decay_parameter))
    plt.ylabel('Overall accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower right")
    plt.show()

# plot all overall accuracies to compare the best batch size
plt.figure()
for decay_parameter in decay:
    plt.plot(overall_accuracy[decay_parameter], label='training accuracy for decay parameter ' + str(decay_parameter))
plt.ylabel('Overall accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

# Plot the train and test accuracies against epochs for the optimal batch size
model = create_model()
histories['optimal'] = model.fit(train_X, train_Y,
                                epochs=epochs,
                                verbose = 2,
                                batch_size=8,
                                validation_data=(test_X,test_Y))
plt.figure()
plt.plot(histories['optimal'].history['accuracy'], label='training accuracy')
plt.plot(histories['optimal'].history['val_accuracy'], label='validation accuracy')
plt.ylabel('Overall accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()
