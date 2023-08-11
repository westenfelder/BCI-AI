from keras import models
import pickle
import numpy as np
from matplotlib import pyplot as plt

NUM_EPOCHS = 10
BATCH_SIZE = 10

with open('training_data.pkl', 'rb') as file:
        training_data = pickle.load(file)

with open('training_labels.pkl', 'rb') as file:
        training_labels = pickle.load(file)

with open('validation_data.pkl', 'rb') as file:
        validation_data = pickle.load(file)

with open('validation_labels.pkl', 'rb') as file:
        validation_labels = pickle.load(file)


network = models.load_model('consistent.h5')

history = network.fit(training_data, training_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(validation_data, validation_labels))

network.save('new.h5')

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