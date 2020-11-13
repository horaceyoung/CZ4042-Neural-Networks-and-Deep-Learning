#
# Project 2, starter code Part a
#

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models

# This is required when using GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


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


def make_model(num_ch_c1, num_ch_c2, use_dropout):
    ''' Note: This model is incomplete. You need to add suitable layers.
    '''

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(3072,)))
    model.add(layers.Reshape(target_shape=(32, 32, 3), input_shape=(3072,)))  # why not just input(shape=(32, 32, 3))?

    model.add(layers.Conv2D(num_ch_c1, 9, activation='relu', input_shape=(None, None, 3)))  # default padding is VALID
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))  # default padding is VALID

    model.add(layers.Conv2D(num_ch_c2, 5, activation='relu', input_shape=(None, None, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(layers.Flatten())

    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(300))

    if use_dropout:
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))
    # Here no softmax because we have combined it with the loss(from_logits=True)

    return model


def main():
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_ch_c1 = 70  # Optimized param from Question 2
    num_ch_c2 = 20  # Optimized param from Question 2

    epochs = 1000  # Fixed
    batch_size = 128  # Fixed
    learning_rate = 0.001
    optimizers = ['SGD', 'SGD-momentum', 'RMSProp', 'Adam', 'dropouts']
    # Question 3 note: the last element 'dropouts' modifies the use_dropout flag and uses SGD optimizer
    use_dropout = False  # Question 3(d) (see make_model)

    # Training and test
    x_train, y_train = load_data('data_batch_1')
    x_test, y_test = load_data('test_batch_trim')

    for optimizer_ in optimizers:
        if optimizer_ == 'SGD':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_ == 'SGD-momentum':  # Question 3(a)
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.1)
        elif optimizer_ == 'RMSProp':  # Question 3(b)
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_ == 'Adam':  # Question 3(c)
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_ == 'dropouts':
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
            use_dropout = True
        else:
            raise NotImplementedError(f'You do not need to handle [{optimizer_}] in this project.')

        # create models
        model = make_model(num_ch_c1, num_ch_c2, use_dropout)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Training
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=2)

        # Create folder to store models and results
        if not os.path.exists('./models'):
            os.mkdir('./models')
        if not os.path.exists('./results'):
            os.mkdir('./results')

        # Save model
        if use_dropout:
            model.save(f'.\models\{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout')
        else:
            model.save(f'.\models\{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout')

        # evaluate on testing data
        score = model.evaluate(x=x_test, y=y_test, verbose=0, batch_size=batch_size)
        print(f'Model {num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout score: {score}')

        # Save the plot for losses
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Test')
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        if use_dropout:
            plt.savefig(
                f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_loss.pdf')
        else:
            plt.savefig(
                f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_loss.pdf'
            )
        plt.close()

        # Save the plot for accuracies
        train_acc = history.history['accuracy']
        test_acc = history.history['val_accuracy']

        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Train')
        plt.plot(range(1, len(test_acc) + 1), test_acc, label='Test')
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        if use_dropout:
            plt.savefig(
                f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_accuracy.pdf'
            )
        else:
            plt.savefig(
                f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_accuracy.pdf'
            )
        plt.close()

if __name__ == '__main__':
    main()
