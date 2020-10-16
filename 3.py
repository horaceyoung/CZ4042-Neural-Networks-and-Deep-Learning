import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

global test_result
histories = {}
test_size = 0.3

def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-X_min)



class TestCallback(tf.keras.callbacks.Callback):
    """
    Generate Test accuracy
    """
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_acc = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.test_acc.append(acc)
        # print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# scale data
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


# build NN model
def build_model(NUM_CLASSES, num_neurons):
    """
    Build the  feedforward neural network
    :param NUM_CLASSES: 3
    :param num_neurons: 5,10,15,20,25
    :return: return a compiled model
    """
    # create model
    model = Sequential()
    model.add(Dense(num_neurons, input_dim=21, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001,
                                      beta_1=0.9,
                                      beta_2=0.999,
                                      epsilon=1e-07,
                                      amsgrad=False
                                      )
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def k_fold_validation(k, NUM_CLASSES, num_neurons, epochs, batch_size):
    """
    Use K-th folder validation to compile the model
    :param k: 5
    :param NUM_CLASSES: 3
    :param num_neurons:  5,10,15,20,25
    :param epochs: 100
    :param batch_size: 4,8,16,32,64
    :return: histories_acc: accuracy of 100 epochs of training data
            histories_loss: loss of 100 epochs of training data
            all_scores: accuracy and loss of validation data,
            model: trained model
    """
    all_scores = []
    histories_acc = []
    histories_loss = []
    histories_acc_test = []
    for i in range(k):
        print('processing fold #', i)
        # prepare validation data: the data from k-th set
        validation_data = trainX[i * num_val_samples: (i + 1) * num_val_samples]
        validation_targets = trainY[i * num_val_samples: (i + 1) * num_val_samples]

        # prepare training data: all the data except k-th set
        partial_train_data = np.concatenate(
            [trainX[:i * num_val_samples],
             trainX[(i + 1) * num_val_samples:]],
            axis=0
        )
        partial_train_target = np.concatenate(
            [trainY[:i * num_val_samples],
             trainY[(i + 1) * num_val_samples:]],
            axis=0
        )

        # create the model
        model = build_model(NUM_CLASSES, num_neurons)

        # Create Callback Class
        test_callback = TestCallback((X_test, y_test))
        # train the model
        history = model.fit(partial_train_data, partial_train_target,
                            epochs=epochs,
                            verbose=0,
                            batch_size=batch_size,
                            callbacks=[test_callback])
        # Training History
        acc_history = history.history['accuracy']
        histories_acc.append(acc_history)

        loss_history = history.history['loss']
        histories_loss.append(loss_history)

        # Test History
        test_acc_history = test_callback.test_acc
        histories_acc_test.append(test_acc_history)

        # evaluate the model on validation_data
        score = model.evaluate(validation_data, validation_targets, verbose=2)
        all_scores.append(score)  # should have 5 histories

    return histories_acc, histories_loss, all_scores, model, histories_acc_test


def plot_train_acc(average_acc_history):
    """
    :param average_acc_history: average score of accuracy 5 set of training data
    :return:
    """
    plt.clf()
    plt.plot(range(1, len(average_acc_history) + 1), average_acc_history)
    plt.xlabel('Epochs')
    plt.ylabel('Training acccuracy')
    plt.savefig('train_acc.png')


def plot_test_acc(average_test_accuracy):
    plt.clf()
    plt.plot(range(1, len(average_test_accuracy) + 1), average_test_accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Test acccuracy')
    plt.savefig('test_acc.png')


if __name__ == '__main__':
    # read train data
    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter=',')
    trainX, trainY = train_input[1:, :21], train_input[1:, -1].astype(int)
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))
    trainY = trainY - 1

    # Split the data randomly into 7:3 training set and test set
    train_X, test_X, train_Y, test_Y = train_test_split(trainX, trainY, test_size=test_size, random_state=1)

    # define parameters
    NUM_CLASSES = 3
    epochs = 100  # Number of epochs
    batch_size = 32  # TODO: Task2 Batch Size Modify Here
    num_neurons = 10 # TODO: Task3 Neurons Modify Here
    seed = 10

    np.random.seed(seed)
    tf.random.set_seed(seed)

    # # compile the model use K-th folder validation
    # k = 5
    # num_val_samples = len(trainX) // k
    # histories_acc, histories_loss, all_scores, model, histories_acc_test = \
    #     k_fold_validation(k, NUM_CLASSES, num_neurons, epochs, batch_size)
    # print(all_scores)
    #
    # # plot train accuracy result
    # average_acc_history = [np.mean([x[i] for x in histories_acc]) for i in range(epochs)]
    # plot_train_acc(average_acc_history)
    #
    # # plot test accuracy result
    # average_test_accuracy = np.array(histories_acc_test).mean(axis=0)
    # plot_test_acc(average_test_accuracy)
    #
    # # predict on test data
    # test_loss, test_acc = model.evaluate(X_test, y_test)
    # print(test_acc)
    # print(histories_acc_test)
    # print(len(histories_acc_test), len(histories_acc_test[0]))

    model = build_model(3, 10)
    model.compile(optimizer='sgd',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train the model
    histories['starter'] = model.fit(train_X, train_Y,
                                     epochs=epochs,
                                     verbose=2,
                                     batch_size=batch_size)

    # plot learning curves
    plt.plot(histories['starter'].history['accuracy'], label=' starter model training accuracy')
    plt.ylabel('Train accuracy')
    plt.xlabel('No. epoch')
    plt.legend(loc="lower right")
    plt.show()
