import threading
import live_graph
import cyton_funcs
import numpy as np
import time
import random
from keras import models
import pygame
import sys

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 0, 221)
FLASH_DELAY_MS = 100 # flash shapes every 300 ms
RUNTIME_SEC = 20 # run the flashing for 20 seconds
STARTUP_DELAY_SEC = 5 # wait 5 seconds before flashing shapes
PORT = 'COM3'
# PORT = '/dev/ttyUSB0'
# sudo chmod a+rw /dev/ttyUSB0


# Note from CAM - You should pay extra attention to this line: network._make_predict_function(), because you have to initialize before threading.
# This is not made clear in the Keras docs, but it's necessary to make the model work concurrently. 
# In short, _make_predict_function is a function that compiles the predict function. 
# In multi thread setting, you have to manually call this function to compile predict in advance, 
#    otherwise the predict function will not be compiled until you run it the first time, which will be problematic when many threads are calling 
#    it at once. You can see a detailed explanation: https://github.com/keras-team/keras/issues/6124

# we will use a thread lock to prevent potential race conditions when updating lists
_lock = threading.Lock()

# function to be called by threads
def prediction_thread(shape, board, channel_nums, sampling_rate, network, prediction_list, averages):
    # sleep 1 second so that get_data() gets the correct data
    time.sleep(1)
    channel_data = cyton_funcs.get_data(board, channel_nums, sampling_rate)
   
    # pass data to nueral network
    prediction_array = network.predict(np.array([channel_data,]), verbose=0)
    prediction = prediction_array[0][0]

    # add shape and prediction to list, basically a history of each prediction
    with _lock: # lock this segment of code so that only one thread at a time may enter
        prediction_list.append([shape, prediction])
        # average the predicton value for given shape
        if shape == 'Circle':
            averages[0] = (averages[0] + prediction) / averages[3]
            averages[3] += 1
        if shape == 'Square':
            averages[1] = (averages[1] + prediction) / averages[4]
            averages[4] += 1
        if shape == 'Triangle':
            averages[2] = (averages[2] + prediction) / averages[5]
            averages[5] += 1


# MAIN FUNCTION
# create list to hold prediction for each flash
prediction_list = []
# list to hold average prediction and prediction count
# averageCirclePrediction, averageSquarePrediction, averageTrianglePrediction, circleFlashCount, squareFlashCount, triangleFlashCount
averages = [0, 0, 0, 1, 1, 1]

# initialize board and model
board, channel_nums, sampling_rate = cyton_funcs.connect_cyton(port=PORT, board_id=0)
network = models.load_model('new.h5')
# network._make_predict_function() # pre-compile the predict function before using it in background threads.

# start live graph in seperate thread
live_graph = threading.Thread(target=live_graph.Graph, args=(board,))
live_graph.start()

time.sleep(3)

# start pygame create white window
pygame.init()
screen = pygame.display.set_mode((1000, 500))
screen.fill(WHITE)

# set delay time
current_time = pygame.time.get_ticks()
delay_ms = FLASH_DELAY_MS
change_time = current_time + delay_ms
flash = False

# pick random shape
shapes = ['Circle', 'Square', 'Triangle']
flash_shape = random.choice(shapes)

# varaible switches when the state changes
current_state = 0

# start display
while current_time < (RUNTIME_SEC * 1000):

    # determine if it is time to flash object
    current_time = pygame.time.get_ticks()
    if current_time >= change_time:
        # update time of next change
        change_time = current_time + delay_ms
        flash = not flash
        flash_shape = random.choice(shapes)

    # flash a shape
    if (flash) & (current_time > (STARTUP_DELAY_SEC * 1000)):
        if flash_shape == 'Circle':
            pygame.draw.circle(screen, BLACK, (100, 250), 50)
        if flash_shape == 'Square':
            pygame.draw.rect(screen, BLACK, (400, 200, 100, 100))
        if flash_shape == 'Triangle':
            pygame.draw.polygon(screen, BLACK, ((800, 200), (900, 200), (850, 300)))
        
        # on each flash start a thread to get prediction, pass in the flashed shape
        if current_state == 1:
            thread = threading.Thread(target=prediction_thread, args=(flash_shape, board, channel_nums, sampling_rate, network, prediction_list, averages,))
            thread.start()
            current_state = 0

    # do not flash any shapes
    if not flash:
        pygame.draw.circle(screen, BLUE, (100, 250), 50)
        pygame.draw.rect(screen, BLUE, (400, 200, 100, 100))
        pygame.draw.polygon(screen, BLUE, ((800, 200), (900, 200), (850, 300)))
        if current_state == 0:
            current_state = 1

    pygame.display.update()
    

# disconnect cyton once live graph is closed
live_graph.join()
time.sleep(2)
cyton_funcs.disconnect_cyton(board)

# print averages and predicted shape
print(averages)
if (averages[0] > averages[1]) & (averages[0] > averages[2]):
    print('Circle')
if (averages[1] > averages[0]) & (averages[1] > averages[2]):
    print('Square')
if (averages[2] > averages[0]) & (averages[2] > averages[1]):
    print('Triangle')
