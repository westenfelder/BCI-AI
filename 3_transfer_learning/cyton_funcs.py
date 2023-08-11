import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# connect to cyton and return board information
def connect_cyton(port, board_id):
    # Connect to Cyton
    params = BrainFlowInputParams()
    params.serial_port = port
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    # Get board information
    channel_nums = board.get_eeg_channels(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    # allow board to initialize
    time.sleep(5)
    return board, channel_nums, sampling_rate

# disconnect from cyton
def disconnect_cyton(board):
    # Disconnect from Cyton
    board.stop_stream()
    board.release_session()

# get data buffer and apply filters
def get_data(board, channel_nums, sampling_rate):
    # get the latest 256 data points
    data = board.get_current_board_data(256)
    channels = data[channel_nums]

    # apply filters
    for i in range(len(channels)):
        # Subtract trend from data (simply removes long-term increase or decrease in the level of the time series.)
        DataFilter.detrend(channels[i], DetrendOperations.CONSTANT.value)
        # Perform band pass filter in-place between 3Hz and 45Hz using 2nd order filter
        DataFilter.perform_bandpass(channels[i], sampling_rate, 3.0, 45.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        # Perform band stop (i.e. notch) filter (inverse of bandpass) between 48Hz and 52Hz using 2nd order filter
        DataFilter.perform_bandstop(channels[i], sampling_rate, 48.0, 52.0, 2, FilterTypes.BUTTERWORTH.value, 0)
        # Perform another band stop filter between 58Hz and 62Hz using 2nd order filter
        DataFilter.perform_bandstop(channels[i], sampling_rate, 58.0, 62.0, 2, FilterTypes.BUTTERWORTH.value, 0)
    return channels
