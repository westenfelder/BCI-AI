# IMPORTS
import numpy as np
from matplotlib import pyplot as plt
from typing import List
import numpy as np
from keras import models
import dataset as api
import random
import time
import scipy.interpolate as interp

# Seed random number generator
random.seed(time.time())

# CONSTANTS
RANDOM_SEED = random.randint(0, 100)

# DEFINE CHANNEL MAP
p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

#LOAD TEST DATA
data = api.create_match_detection_dataset(data_pickle_file='data.dat', channel_map=p300_8_channel_map, random_seed=RANDOM_SEED)
validation_data = data[2]
validation_labels = data[3]

# LOAD MODEL AND GET PREDICTION
network = models.load_model('consistent.h5')
random_subject = random.randint(0, len(validation_data)) 
guess = network.predict(np.array([validation_data[random_subject],]))

# # GRAPH DATA
# # strech 256 samples over 1000 ms window
# def channel(j):
#     y = np.asarray(validation_data[random_subject][j])
#     y_inter = interp.interp1d(np.arange(y.size), y)
#     y_ = y_inter(np.linspace(0,y.size-1,x.size))
#     return y_

# x = np.arange(1000)
# # plot each channel
# for i in range(validation_data.shape[1]):
#     plt.plot(x, channel(i), label='channel ' + str(i))

# # set axis and title 
# plt.axvline(x=300)
# plt.title(f"Match: {str(validation_labels[random_subject])}  Prediction: {str(guess[0][0])}")
# plt.xlabel('Miliseconds') 
# plt.ylabel('Volts')
# plt.legend()
# plt.show()

def plot_all_channels(multiChannels, title ='EEG Channel Plot', chnl_label='channel'):
    # x-axis is spread accross 1000 ms
    t = np.linspace(0.0, 1000, len(multiChannels[0]))
    plt.figure()

    # Plot all channels onto the same canvas/window
    n=1
    for chnl in multiChannels:
        # plot current channel
        plt.subplot(1, 1, 1)
        plt.plot(t, chnl, label =chnl_label + " " + str(n), linewidth=1)
        n+=1

    plt.title(title)
    plt.ylabel('Amplitude (uV)')
    plt.xlabel('Time (ms)')
    plt.legend(loc='lower right')
    plt.pause(0.01)

    # Display interactive plot window
    plt.show(block = False)
    plt.pause(0.01)

plot_all_channels(validation_data[random_subject])