from keras import Input, Model
from keras.layers import Dense
import numpy as np

train_data = np.load('dataset1.npy')
value_data = np.load('outcome_ohv_r5_1.npy')

encoding_dim1 = 20  # size of encoded representations
encoding_dim2 = 10  # size of encoded representations in the bottleneck layer
data_dim = train_data.shape[1]
encoder_epochs = 250
encoder_batchsize = 10
n_epochs = 250
n_batchsize = 10

# define the auto-encoder and train it
# this is our input placeholder
input_data = Input(shape=train_data[0].shape)
# "encoded" is the first encoded representation of the input
encoded = Dense(encoding_dim1, activation='relu', name='encoder1')(input_data)
# "enc" is the second encoded representation of the input
enc = Dense(encoding_dim2, activation='relu', name='encoder2')(encoded)
# "dec" is the lossy reconstruction of the input
dec = Dense(encoding_dim1, activation='sigmoid', name='decoder1')(enc)
# "decoded" is the final lossy reconstruction of the input
decoded = Dense(data_dim, activation='sigmoid', name='decoder2')(dec)

ae = Model(inputs=input_data, outputs=decoded)  # autoencoder model ae
ae.compile(optimizer='sgd', loss='mse')
ae.fit(train_data[:10000], train_data[:10000],
       epochs=encoder_epochs,
       batch_size=encoder_batchsize,
       shuffle=True,
       validation_data=(train_data[10001:13000], train_data[10001:13000]))

# create a MLP classifier
# create model
# get autoencoder's output as the iput of MLP
x = ae.get_layer('encoder2').output
h1 = Dense(128, activation='relu', name='hidden1')(x)
h2 = Dense(256, activation='relu', name='hidden2')(h1)  # hidden layers
h3 = Dense(512, activation='relu', name='hidden3')(h2)
y = Dense(5, activation='softmax', name='predictions')(h3)
classifier = Model(inputs=ae.inputs, outputs=y)
classifier.summary()

# Compile model
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
classifier.fit(train_data[:10000], value_data[:10000], batch_size=n_batchsize, epochs=n_epochs, verbose=1,
               validation_data=(train_data[10001:13000], value_data[10001:13000]))

# evaluate model
score = classifier.evaluate(train_data[13001:15000], value_data[13001:15000], verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
