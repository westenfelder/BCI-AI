# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
from keras import models

#LOAD TEST DATA
test = np.load('test_images.pkl', allow_pickle=True)

# LOAD MODEL
network = models.load_model('model.h5')

#TEST NETWORK
for i in range(len(test)):
    guess = network.predict(np.array([test[i],]))
    if guess[0][0] < 0.5:
        print("Cat")
    else:
        print("Dog")

    plt.imshow(test[i])
    plt.show()