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
    model.add(layers.Input(shape=(3072, )))
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
    model.add(layers.Dense(10, use_bias=True, input_shape=(300,)))  # Here no softmax because we have combined it with the loss(from_logits=True)
    
    return model


def main():
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_ch_c1 = 50  # Question 2
    num_ch_c2 = 60  # Question 2

    # epochs = 1000  # Fixed
    epochs = 10  # just a trial
    batch_size = 128  # Fixed
    learning_rate = 0.001
    optimizer_ = 'SGD'  # Question 3
    use_dropout = False  # Question 3(d) (see make_model)

    model = make_model(num_ch_c1, num_ch_c2, use_dropout)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if optimizer_ == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_ == 'SGD-momentum':  # Question 3(a): SGD with momentum = 0.1
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'RMSProp':  # Question 3(b): RMSProp
        raise NotImplementedError('Complete it by yourself')
    elif optimizer_ == 'Adam':  # Question 3(c): Adam
        raise NotImplementedError('Complete it by yourself')
    else:
        raise NotImplementedError(f'You do not need to handle [{optimizer_}] in this project.')

    # Training and test
    x_train, y_train = load_data('data_batch_1')
    x_test, y_test = load_data('test_batch_trim')

    # Training
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test))

    ''' Fill in Question 1(b) here. This website may help:
            https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
    '''
    layer_outputs = [layer.output for layer in model.layers[1:5]]
    activation_model = models.Model(model.input, layer_outputs)

    test_imgs = x_test[0:2]  
    activations = activation_model.predict(test_imgs)  # [(2, 24, 24, 50)]

    for i, img in enumerate(test_imgs):
        img = tf.reshape(img, [32, 32, 3])
        plt.imshow(img, interpolation='nearest')
        plt.savefig(f'./results/testImage_{i}.png')

    layer_names = [layer.name for layer in model.layers[1:5]]
    images_per_row = 10


    for img_idx in [0, 1]:
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size*n_cols, images_per_row*size))
            
            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_img = layer_activation[img_idx, :, :, col*images_per_row + row]
                    channel_img -= channel_img.mean()
                    channel_img /= channel_img.std()
                    channel_img *= 64
                    channel_img += 128
                    channel_img = np.clip(channel_img, 0, 255).astype('uint8')
                    display_grid[col*size:(col+1)*size, row*size:(row+1)*size] = channel_img
            scale = 1./size
            plt.figure(figsize=(scale*display_grid.shape[1],
                                scale*display_grid.shape[0]))
            plt.title("Image "+ str(img_idx) + ", "+ layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto')
            plt.savefig(
                f'./results/Image_{img_idx}_{layer_name}.png')
            plt.clf()

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
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_loss.png')
    else:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_loss.png')
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
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_dropout_accuracy.png'
        )
    else:
        plt.savefig(
            f'./results/{num_ch_c1}_{num_ch_c2}_{optimizer_}_no_dropout_accuracy.png'
        )
    plt.close()


if __name__ == '__main__':
    main()
