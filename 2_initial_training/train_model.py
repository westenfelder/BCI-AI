# IMPORTS
from typing import List
import numpy as np
from matplotlib import pyplot as plt
from keras import models
from keras import layers
import dataset as api
import random
import time

# Seed random number generator
random.seed(time.time())

# CONSTANTS
NUM_EPOCHS = 40
RANDOM_SEED = random.randint(0, 100)
BATCH_SIZE = 10

# DEFINE CHANNEL MAP
p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

# LOAD DATA
data = api.create_match_detection_dataset(data_pickle_file='data.dat', channel_map=p300_8_channel_map, random_seed=RANDOM_SEED)
training_data = np.concatenate((data[0], data[4]), axis=0)
training_labels = np.concatenate((data[1], data[5]), axis=0)
validation_data = data[2]
validation_labels = data[3]

# BUILD NETWORK
network = models.Sequential()
# Convolutional layers
network.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', input_shape=(8, 256), data_format='channels_first'))
network.add(layers.Conv1D(filters=64, kernel_size=7, padding='same'))
network.add(layers.Conv1D(filters=32, kernel_size=11, padding='same'))
network.add(layers.MaxPool1D(pool_size=2))
network.add(layers.Flatten())
# Dense layer 1, normalized
network.add(layers.Dense(units=1024))
network.add(layers.LayerNormalization())
network.add(layers.ELU())
# Dense layer 2
network.add(layers.Dense(units=512))
# Dense layer 3
network.add(layers.Dense(units=256))
# Dense layer 4, normalized
network.add(layers.Dense(units=128))
network.add(layers.LayerNormalization())
network.add(layers.ELU())
# Final layer, binary classifier
network.add(layers.Dense(units=1, activation='sigmoid'))


# DEFINE OPTIMIZER AND LOSS FUNCTION
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# display network details
network.summary()

# TRAIN THE NETWORK
history = network.fit(training_data, training_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(validation_data, validation_labels))

# SAVE THE NETWORK
network.save('model.h5')

# GRAPH ACCURACY 
acc = np.asarray(history.history['accuracy'])
val_acc = np.asarray(history.history['val_accuracy'])
loss = np.asarray(history.history['loss']) 
val_loss = np.asarray(history.history['val_loss'])
epochs = np.asarray(range(len(acc)))
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.plot(epochs, loss, label='Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training/Validation Accuracy and Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy and Loss') 
plt.legend() 
plt.show()