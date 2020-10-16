#
# Project 1, starter code part a
#
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)

# constants
NUM_CLASSES = 3

epochs = 5000
batch_size = 32
num_neurons = 10
seed = 10
test_size = 0.3

np.random.seed(seed)
tf.random.set_seed(seed)

histories = {}

#read train data
train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter= ',')
trainX, train_Y = train_input[1:, :21], train_input[1:,-1].astype(int)
trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
trainY = train_Y-1

# Split the data randomly into 7:3 training set and test set
train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size = test_size, random_state=1)

# create the model
model = keras.Sequential()
model.add(keras.layers.Dense(num_neurons, input_dim=21, activation='relu',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax',
                             kernel_initializer='random_normal',
                             bias_initializer='zeros'
                             ))
optimizer = keras.optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )


# train the model
class MetricCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print("For epoch {}, testing loss is {:7.2f}.".format(epoch, logs["val_loss"]))


histories['training_model'] = model.fit(train_X, train_Y,
                                         epochs=epochs,
                                         verbose = 0,
                                         batch_size=batch_size,
                                         validation_data=(test_X, test_Y),
                                         callbacks=[MetricCallback()])

# plot learning accuracy curves
plt.figure()
plt.plot(histories['training_model'].history['accuracy'], 'g', label='model training accuracy')
plt.plot(histories['training_model'].history['val_accuracy'], 'r', label='model validation accuracy')
plt.ylabel('Train accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()

plt.figure()
plt.plot(histories['training_model'].history['loss'], 'g', label='model training loss')
plt.plot(histories['training_model'].history['val_loss'], 'r', label='model validation loss')
plt.ylabel('Train loss')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()



