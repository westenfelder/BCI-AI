# python -m pip install -U brainflow
# pip3 install PyQt5
# python -m pip install pyqtgraph==0.12.1
# program will only work with ^ version of pyqtgraph

from brainflow.board_shim import BoardShim
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

class Graph:

    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4  # windows size in seconds

        # Determine the amount of data we will read at once from board's buffer.
        self.num_points = self.window_size * self.sampling_rate

        # Note that app keeps program alive until graph window is closed
        self.app = QtGui.QApplication([])

        # Create window to hold realtime graph.
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(800, 600))

        # Now, setup/configure the graph to hold 8 channels of data
        self._init_timeseries()

        # Setup callback timer responsible for prompting
        #  us when to read the next chunk of data from the board.
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)

        # Now, tell that app to stay alive (until the user closes the window).
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):

        # Prepare graph to hold 8 channels of streaming data.
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    # Note that our Timer will call back into this function every 50ms
    def update(self):

        # Read data from the board

        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # Plot timeseries...

            # First, apply data filters. Learn more with link below.
            # https://brainflow.readthedocs.io/en/stable/UserAPI.html?highlight=Datafilter#_CPPv410DataFilter

            # Subtract trend from data (simply removes long-term increase or decrease in the level of the time series.)
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)

            # Perform band pass filter in-place between 3Hz and 45Hz using 2nd order filter
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            # Perform band stop (i.e. notch) filter (inverse of bandpass) between 48Hz and 52Hz using 2nd order filter
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            # Perform another band stop filter between 58Hz and 62Hz using 2nd order filter
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)

            # Place processed data into the graph's buffer
            self.curves[count].setData(data[channel].tolist())

        self.app.processEvents()