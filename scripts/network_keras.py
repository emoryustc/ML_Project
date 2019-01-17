from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

train_data = np.load('dataset.npy')
value_data = np.load('outcome_ohv_r5.npy')

input_dim = train_data.shape[1]
output_dim = nb_classes = 5
batch_size = 128
nb_epoch = 100


def create_network_0():
    """
    Create simple network

    :return:
    """
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
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
    model = create_network_0()
    train_model(model)
    evaluate_model(model)
