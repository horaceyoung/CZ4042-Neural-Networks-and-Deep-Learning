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

# create a network
def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(num_neurons, input_dim=7, activation='relu',
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
# learn the network
model = create_model()
histories['model'] =model.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 2,
                                        validation_data=(test_X, test_Y))

print(histories['model'].history.keys())
# plot learning curves
plt.figure()
plt.plot(histories['model'].history['mse'], label='model training mse')
plt.plot(histories['model'].history['val_mse'], label='model validation mse')
plt.ylabel('mse')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
early_stop_model = create_model()
histories['model_early_stopping'] = early_stop_model.fit(train_X, train_Y,
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose = 2,
                                        validation_data=(test_X, test_Y),
                                        callbacks=callback)
plt.figure()
plt.plot(histories['model_early_stopping'].history['mse'], label='model training mse')
plt.plot(histories['model_early_stopping'].history['val_mse'], label='model validation mse')
plt.ylabel('mse')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

# Split the data randomly into 1:7, we will use predict_x to evaluate the model accuracy
rest_X, predict_X, rest_Y, predict_Y = train_test_split(trainX, trainY, test_size = 0.125, random_state=2)
result = model.predict(predict_X)
plt.figure()
x = range(1, 51)
print(len(result))
print(len(x))
plt.scatter(x, result, label='model predict result')
plt.scatter(x, predict_Y, label='actural result')
plt.ylabel('value')
plt.xlabel('records')
plt.legend(loc="lower right")
plt.show()