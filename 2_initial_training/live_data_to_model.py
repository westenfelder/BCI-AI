import time
import matplotlib.pyplot as plt
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from scipy.signal import savgol_filter
import numpy as np
from typing import List
from keras import models
import dataset as api
import random
import time
import pickle
import scipy.interpolate as interp

# Connection Options
PORT = 'COM3'
BOARD_ID = 0

# Sampling options
RUN_TIME = 1 # Run time in seconds
SAMPLES_PER_SECOND = 10 # Number of times to collect data from buffer
DATA_RATE = 250 # 250hz, sends data to buffer every .5 seconds
BUFFER_SIZE = 256

# Graph options
DISPLAY_DATA_SEGMENTS = True
DISPLAY_ALL_DATA = False
CHANNEL_NUM = 3

# Connect to Cyton
params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BOARD_ID, params)
board.prepare_session()
board.start_stream()
channel_nums = board.get_eeg_channels(BOARD_ID)
time.sleep(5) # allow board to initialize

# convert microvolts to volts
def uVtoV(num):
    return num / 1e6
uVtoVarray = np.vectorize(uVtoV)

# Collect data from board
data = board.get_current_board_data(BUFFER_SIZE) # get the latest 256 data points
channels = data[channel_nums] # get channel data

# dump collected data
with open('live_data_example.dat', 'wb') as f:
    pickle.dump(data, f)

print(channels[0])
print()
channels = uVtoVarray(channels)
print(channels[0])

# Disconnect from Cyton
board.stop_stream()
board.release_session()

# LOAD MODEL AND GET PREDICTION
network = models.load_model('consistent.h5')
guess = network.predict(np.array([channels,]))

# GRAPH DATA
# strech 256 samples over 1000 ms window
def strech(j):
    y = np.asarray(channels[j])
    y_inter = interp.interp1d(np.arange(y.size), y)
    y_ = y_inter(np.linspace(0,y.size-1,x.size))
    return y_

x = np.arange(1000)
# plot each channel
for i in range(channels.shape[0]):
    plt.plot(x, strech(i), label='channel ' + str(i))

# set axis and title 
plt.axvline(x=300)
plt.title(f"Prediction: {str(guess[0][0])}")
plt.xlabel('Miliseconds') 
plt.ylabel('Volts')
plt.legend()
plt.show()
