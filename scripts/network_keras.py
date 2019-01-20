from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf

import numpy as np

train_data = np.load('dataset1.npy')
value_data = np.load('outcome_ohv_r5_1.npy')

input_dim = train_data.shape[1]
output_dim = 5
batch_size = 256
nb_epoch = 200


def create_network_0():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    model.summary()
    return model


def create_network_1():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=128, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_2():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=256, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_3():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(64, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=256, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_4():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=512, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_5():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(1024, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=1024, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_6():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=512, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_7():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(1024, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(1024, input_dim=1024, activation=tf.nn.relu))
    model.add(Dense(1024, input_dim=1024, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=1024, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_8():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=512, activation=tf.nn.relu))
    model.add(Dropout(0.7, noise_shape=None, seed=None))
    model.add(Dense(1024, input_dim=512, activation=tf.nn.relu))
    model.add(Dropout(0.7, noise_shape=None, seed=None))
    model.add(Dense(1024, input_dim=1024, activation=tf.nn.relu))
    model.add(Dropout(0.7, noise_shape=None, seed=None))
    model.add(Dense(1024, input_dim=1024, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=1024, activation=tf.nn.softmax))
    model.summary()
    return model


def create_network_9():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation=tf.nn.relu))
    model.add(Dense(256, input_dim=128, activation=tf.nn.relu))
    model.add(Dense(512, input_dim=256, activation=tf.nn.relu))
    model.add(Dense(1024, input_dim=512, activation=tf.nn.relu))
    model.add(Dense(2048, input_dim=1024, activation=tf.nn.relu))
    model.add(Dense(4096, input_dim=2048, activation=tf.nn.relu))
    model.add(Dense(output_dim, input_dim=4096, activation=tf.nn.softmax))
    model.summary()
    return model


def train_model(model):
    """
    Train the network

    :param model:
    :return:
    """
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data[:10000], value_data[:10000], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
              validation_data=(train_data[10001:13000], value_data[10001:13000]))


def evaluate_model(model):
    """
    Evaluate model

    :param model:
    :return:
    """
    score = model.evaluate(train_data[13001:15000], value_data[13001:15000], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    my_model = create_network_9()
    train_model(my_model)
    evaluate_model(my_model)
