import threading
import live_graph
import cyton_funcs
import time
import random
import pygame
import pickle

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 0, 221)
FLASH_DELAY_MS = 300 # flash shapes every 300 ms
RUNTIME_SEC = 30 # run the flashing for 20 seconds
STARTUP_DELAY_SEC = 10 # wait 5 seconds before flashing shapes
# PORT = 'COM3'
PORT = '/dev/ttyUSB0'


# function to be called by threads
def prediction_thread(bool, board, channel_nums, sampling_rate, data_list):
    # sleep 1 second so that get_data() gets the correct data
    time.sleep(1)
    channel_data = cyton_funcs.get_data(board, channel_nums, sampling_rate)

    # add shape and prediction to list, basically a history of each prediction
    data_list.append([bool, channel_data])


# MAIN FUNCTION
# create list to hold data for each flash
data_list = []

# initialize board
board, channel_nums, sampling_rate = cyton_funcs.connect_cyton(port=PORT, board_id=0)

# start live graph in seperate thread
# live_graph = threading.Thread(target=live_graph.Graph, args=(board,))
# live_graph.start()

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
look_shape = random.choice(shapes)
print("Look at " + look_shape)

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
        
        if current_state == 1:
            current_state = 0
            # flashes shape user is looking at
            if flash_shape == look_shape:
                thread = threading.Thread(target=prediction_thread, args=(True, board, channel_nums, sampling_rate, data_list,))
                thread.start()
            # flashes differnt shape
            else:
                thread = threading.Thread(target=prediction_thread, args=(False, board, channel_nums, sampling_rate, data_list,))
                thread.start()
            

    # do not flash any shapes
    if not flash:
        pygame.draw.circle(screen, RED, (100, 250), 50)
        pygame.draw.rect(screen, GREEN, (400, 200, 100, 100))
        pygame.draw.polygon(screen, BLUE, ((800, 200), (900, 200), (850, 300)))
        if current_state == 0:
            current_state = 1

    pygame.display.update()
    

# disconnect cyton once live graph is closed
# live_graph.join()
time.sleep(2)
cyton_funcs.disconnect_cyton(board)

data = []
labels = []

for i in range(len(data_list)):
    labels.append(data_list[i][0])
    data.append(data_list[i][1])

with open('data.pkl', 'wb') as file:
        pickle.dump(data, file)

with open('labels.pkl', 'wb') as file:
        pickle.dump(labels, file)