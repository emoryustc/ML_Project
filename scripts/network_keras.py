from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
train_data = np.load('dataset.npy')
value_data = np.load('outcome_ohv.npy')
print(train_data.shape)
print(value_data.shape)

input_dim = train_data.shape[1]

output_dim = nb_classes = 5

model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
batch_size = 128
nb_epoch = 20

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data[:10000], value_data[:10000], batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(train_data[10001:13000], value_data[10001:13000]))
score = model.evaluate(train_data[13001:15000], value_data[13001:15000], verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
