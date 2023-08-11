# INSTALL KERAS
# pip install tensorflow-cpu
# pip install keras

# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
from keras import models
from keras import layers
from keras import regularizers

# LOAD DATA
images = np.load('images.pkl', allow_pickle=True)
labels = np.load('labels.pkl', allow_pickle=True)

# BUILD NETWORK
network = models.Sequential()
# conv layer 1
# 80, 80, 3 corresponds to image shape
network.add(layers.Conv2D(128,(3,3),input_shape=(80,80,3))) 
network.add(layers.LeakyReLU())
network.add(layers.MaxPooling2D(pool_size=(2,2)))
# conv layer 2
network.add(layers.Conv2D(80,(3,3)))
network.add(layers.LeakyReLU())
network.add(layers.MaxPooling2D(pool_size=(2,2)))
# conv layer 3
network.add(layers.Conv2D(32,(3,3)))
network.add(layers.LeakyReLU())
network.add(layers.MaxPooling2D(pool_size=(2,2)))
# flatten layers
network.add(layers.Flatten())
# intermediate layer
network.add(layers.Dense(128,  kernel_regularizer=regularizers.l2(0.001)))
network.add(layers.LeakyReLU())
# final layer
# final layer has 1 node, either 0 or 1 for cat or dog
network.add(layers.Dense(1,activation='sigmoid')) 
# DEFINE OPTIMIZER AND LOSS FUNCTION
network.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

# TRAIN THE NETWORK
history = network.fit(images,labels,epochs=20,batch_size=10,validation_data=(images,labels))

# SAVE THE NETWORK
network.save('model.h5')

# GRAPH ACCURACY 
acc = np.asarray(history.history['accuracy'])
val_acc = np.asarray(history.history['val_accuracy']) 
epochs = np.asarray(range(len(acc)))
plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='validation accuracy')
plt.title('Training and accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()