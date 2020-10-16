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
num_neurons = 10
seed = 10
test_size = 0.3
features = ['GRE', 'TOEFL', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
optimal_features = ['GRE', 'TOEFL', 'SOP', 'LOR', 'CGPA', 'Research']
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
trainX = X_data
trainY = Y_data

# Split the data randomly into 7:3 training set and test set
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size = test_size, random_state=1)
train_X = (train_X - np.mean(train_X, axis=0)) / np.std(train_X, axis=0)
test_X = (test_X - np.mean(test_X, axis=0)) / np.std(test_X, axis=0)
# removing university rating to acquire the optimal training set
train_X = np.delete(train_X, 2, axis=1)
test_X = np.delete(test_X, 2, axis=1)

# create 4 layer model without dropouts and record the mse
four_layer_model_without_dropouts = keras.Sequential()
four_layer_model_without_dropouts.add(keras.layers.Dense(50, input_dim=6, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
four_layer_model_without_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
four_layer_model_without_dropouts.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

four_layer_model_without_dropouts.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
histories['four_layer_model_without_dropouts'] = four_layer_model_without_dropouts.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 0,
                                        validation_data=(test_X, test_Y))

four_layer_model_without_dropouts_eval = four_layer_model_without_dropouts.evaluate(test_X, test_Y)
print("Four layer model without dropouts evaluation result: " + str(four_layer_model_without_dropouts_eval))


# create 4 layer model with dropouts and record the mse
four_layer_model_with_dropouts = keras.Sequential()
four_layer_model_with_dropouts.add(keras.layers.Dense(50, input_dim=6, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
four_layer_model_with_dropouts.add(keras.layers.Dropout(0.2))
four_layer_model_with_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
four_layer_model_with_dropouts.add(keras.layers.Dropout(0.2))
four_layer_model_with_dropouts.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

four_layer_model_with_dropouts.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
histories['four_layer_model_with_dropouts'] = four_layer_model_with_dropouts.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 0,
                                        validation_data=(test_X, test_Y))

four_layer_model_with_dropouts_eval = four_layer_model_with_dropouts.evaluate(test_X, test_Y)
print("Four layer model with dropouts evaluation result: " + str(four_layer_model_with_dropouts_eval))

# create 5 layer model without dropouts and record the mse
five_layer_model_without_dropouts = keras.Sequential()
five_layer_model_without_dropouts.add(keras.layers.Dense(50, input_dim=6, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_without_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_without_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_without_dropouts.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

five_layer_model_without_dropouts.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
histories['five_layer_model_without_dropouts'] = five_layer_model_without_dropouts.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 0,
                                        validation_data=(test_X, test_Y))

five_layer_model_without_dropouts_eval = five_layer_model_without_dropouts.evaluate(test_X, test_Y)
print("five layer model without dropouts evaluation result: " + str(five_layer_model_without_dropouts_eval))


# create 5 layer model with dropouts and record the mse
five_layer_model_with_dropouts = keras.Sequential()
five_layer_model_with_dropouts.add(keras.layers.Dense(50, input_dim=6, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_with_dropouts.add(keras.layers.Dropout(0.2))
five_layer_model_with_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_with_dropouts.add(keras.layers.Dropout(0.2))
five_layer_model_with_dropouts.add(keras.layers.Dense(50, activation='relu',
                                 kernel_initializer='random_normal',
                                 bias_initializer='zeros',
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
five_layer_model_with_dropouts.add(keras.layers.Dropout(0.2))
five_layer_model_with_dropouts.add(keras.layers.Dense(1))
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

five_layer_model_with_dropouts.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mse']
                  )
histories['five_layer_model_with_dropouts'] = five_layer_model_with_dropouts.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 0,
                                        validation_data=(test_X, test_Y))

five_layer_model_with_dropouts_eval = five_layer_model_with_dropouts.evaluate(test_X, test_Y)
print("five layer model with dropouts evaluation result: " + str(five_layer_model_with_dropouts_eval))



plt.figure()
plt.plot(histories['four_layer_model_without_dropouts'].history['mse'], label='four layer model without dropouts training mse')
plt.plot(histories['four_layer_model_with_dropouts'].history['mse'], label='four layer model with dropouts training mse')
plt.plot(histories['five_layer_model_without_dropouts'].history['mse'], label='five layer model without dropouts training mse')
plt.plot(histories['five_layer_model_with_dropouts'].history['mse'], label='five layer model with dropouts training mse')
plt.ylabel('mse')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()
