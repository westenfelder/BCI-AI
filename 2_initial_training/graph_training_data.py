from typing import List
import numpy as np
from matplotlib import pyplot as plt
import dataset as api
import scipy.interpolate as interp
import random
import time

# show eight channels or all channels
EIGHT_CHANNELS = True

# seed random number generator
random.seed(time.time())

# DEFINE CHANNEL MAP
p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

# LOAD DATA
if EIGHT_CHANNELS:
    data = api.create_match_detection_dataset(data_pickle_file='data.dat', channel_map=p300_8_channel_map, random_seed=random.randint(0, 100))
else:
    data = api.create_match_detection_dataset(data_pickle_file='data.dat', random_seed=random.randint(0, 100))

training_data = data[0]
training_labels = data[1]

# select random trial
random_subject = random.randint(0, len(training_data)) 

# strech 256 samples over 1000 ms window
def channel(j):
    y = np.asarray(training_data[random_subject][j])
    y_inter = interp.interp1d(np.arange(y.size), y)
    y_ = y_inter(np.linspace(0,y.size-1,x.size))
    return y_

x = np.arange(1000)

# plot all channels
for i in range(training_data.shape[1]):
    plt.plot(x, channel(i), label='channel ' + str(i))

# set title and axis
plt.axvline(x=300)
plt.title('Match: ' + str(training_labels[random_subject]))
plt.xlabel('Miliseconds') 
plt.ylabel('Volts')
plt.legend()
plt.show()
