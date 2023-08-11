from typing import List
import dataset as api
import pickle

# define channel map
p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']
# open data folder
subjects = api.import_subjects('../../EEG_Alcoholic/')
# pickle data
with open('data.dat', 'wb') as f:
    pickle.dump(subjects, f)
# for testing
data = api.create_match_detection_dataset(data_pickle_file='data.dat', channel_map=p300_8_channel_map, random_seed=4)