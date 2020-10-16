#
# Project 1, starter code part b
#

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

NUM_CLASSES = 7

epochs = 1000
batch_size = 8
num_neurons = 30
seed = 10
test_size = 0.3
features = ['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
features_remove_UR = ['GRE', 'TOEFL', 'SOP', 'LOR', 'CGPA', 'Research']
histories={}

np.random.seed(seed)
tf.random.set_seed(seed)

#read and divide data into test and train sets 
admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
Y_data = Y_data.reshape(Y_data.shape[0], 1)

idx = np.arange(X_data.shape[0])
np.random.shuffle(idx)
X_data, Y_data = X_data[idx], Y_data[idx]

# Scale the features
trainX = (X_data- np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
trainY = Y_data

# Split the data randomly into 7:3 training set and test set
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size = test_size, random_state=1)


# create a network
def create_model(input_dimension):
    model = keras.Sequential()
    model.add(keras.layers.Dense(num_neurons, input_dim=input_dimension, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(10e-3)))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=10e-3)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
    return model
'''
# learn the network, each time removing a column
input_dim = 6
for i in range(7):
    # remove the ith column
    print("removed feature " + features[i] + " for training:")
    new_train_X = np.delete(train_X, i, axis=1)
    new_test_X = np.delete(test_X, i, axis=1)

    model = create_model(input_dim)
    histories['model'] =model.fit(new_train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 2,
                                        validation_data=(new_test_X, test_Y))

    eval = model.evaluate(new_test_X, test_Y)
    print(eval)

    # plot learning curves
    plt.figure()
    plt.plot(histories['model'].history['mse'], label='model training mse')
    plt.plot(histories['model'].history['val_mse'], label='model validation mse')
    plt.ylabel('mse')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower right")
    plt.show()
'''

# after the code above, we decided to remove university rating
input_dim = 5
train_X_rm_UR = np.delete(train_X, 2, axis=1)
test_X_rm_UR = np.delete(test_X, 2, axis=1)
for i in range(6):
    # remove the ith column
    print("removed feature " + features_remove_UR[i] + " for training:")
    new_train_X = np.delete(train_X_rm_UR, i, axis=1)
    new_test_X = np.delete(test_X_rm_UR, i, axis=1)
    print(train_Y.shape)
    model = create_model(input_dim)
    histories['model'] =model.fit(new_train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 2,
                                        validation_data=(new_test_X, test_Y))

    eval = model.evaluate(new_test_X, test_Y)
    print(eval)

    # plot learning curves
    plt.figure()
    plt.plot(histories['model'].history['mse'], label='model training mse')
    plt.plot(histories['model'].history['val_mse'], label='model validation mse')
    plt.ylabel('mse')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower right")
    plt.show()