from typing import List

import numpy as np
import os
import glob
import pickle
import random
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score
import baker

from IPython import embed

data_path_chad = 
pickle_data_path = 


class cotrial:
    """Holds 1-second trial data"""

    def __init__(self, match_cond, trial_num, epoch_len=3.9):
        self.match_cond = match_cond  # S1 obj, S2 match, or S2 nomatch
        self.epoch_len = epoch_len  # milliseconds
        self.trial_num = trial_num
        self.data = []  # shape (64,256) channels, samples
        self.label = -1


class cosubject:
    """EEG subject class, holds all trial data for subject"""

    def __init__(self, subject, description=None, sr=256):
        self.name = subject  # something like: co2c0000339.rd
        self.sample_rate = sr  # should be 256 (hz)
        self.trials = list()  # holds list of trials/epochs
        self.alcoholic = subject[3] == 'a'
        self.description = description  # 120 trials, 64 chans, 416 samples 368 post_stim samples

    def set_trial_data(self, data):

        if self.alcoholic:
            label_val = 1
        else:
            label_val = 0

        for trial in data:
            trial.label = label_val

        self.trials = data

    def set_erp_trial_data(self, data):
        keep = []
        for trial in data:
            if trial.match_cond == 'S2 nomatch':
                trial.label = 1
                keep.append(trial)
            elif trial.match_cond == 'S2 nomatch':
                trial.label = 0
                keep.append(trial)

        self.trials = keep


def import_subjects(root_folder,
                    track_missing=False,
                    max_trials_per_subject=120,
                    sample_count=256,  # 256 samples per channel
                    channel_count=64,
                    num_lines_per_trial=16452):
    print("Importing all folders under " + root_folder + "...")
    subjects = []
    # process all immediate directories from root...
    for dirs in os.walk(root_folder, topdown=True):
        for d in dirs[1]:
            try:
                print("  Processing subject: " + d + "...")
                os.chdir(root_folder + '/' + d)
                subject = cosubject(d + ".rd")
                subjects.append(subject)

                for trial_num in range(max_trials_per_subject):
                    fsearch = "*.rd." + str(trial_num).zfill(3)
                    flist = glob.glob(fsearch)
                    if len(flist) > 0:

                        for f in flist:
                            print("     .......", f)
                            try:
                                # These files are between 260k and 300k,
                                # so, we'll read all the lines into memory and then parse.
                                f = open(f, "r")
                                lines = list(f)
                                f.close()

                                if len(lines) < num_lines_per_trial:
                                    print("     ********** File not what expected...")
                                    print("     ********** Lines in file: ", len(lines))
                                    print("     ********** Skipping file ", f)
                                    continue

                                # process trial "header" (first 4 lines)
                                if subject.description is None:
                                    subject.description = lines[1].strip()

                                trial_info = lines[3].split(',')
                                trial_match_cond = trial_info[0][2:].strip()
                                epoch = float(lines[2][2:7])
                                trial = cotrial(match_cond=trial_match_cond,
                                                trial_num=trial_num,
                                                epoch_len=epoch)

                                # now, let's load trial data...
                                l_index = 5  # starton this line to get samples
                                data = np.zeros((channel_count, sample_count), float)
                                c_index = 0  # channel index

                                # now, parse data...
                                while l_index < num_lines_per_trial:  # number of lines per trial
                                    for si in range(0, sample_count):
                                        data[c_index][si] = float(lines[l_index].split(' ')[3])
                                        l_index += 1

                                    l_index += 1  # skip over channel header
                                    c_index += 1

                                # add trial to subject
                                trial.data = data
                                subject.trials.append(trial)

                            except Exception as ex:
                                print("     ********** Problem while processing file ", f)
                                print("     ********** Error: ", ex)
                    else:
                        if track_missing:
                            subject.trials.append(None)  # place holder to indicate no trial exists

            except Exception as ex:
                print("Problem while processing directory ", d)
                print("Error: ", ex)

    return subjects


train_partition = [35, 5, 46, 37, 84, 69, 113, 96, 71, 48, 18, 43, 8, 28, 78, 111, 65, 54, 23, 91, 98, 110, 7,
                   26, 17, 63, 55, 16, 95, 120, 52, 99, 97, 105, 116, 60, 34, 94, 29, 93, 45, 1, 10, 51, 47, 74,
                   13, 33, 0, 4, 102, 57, 64, 83, 66, 19, 107, 89, 70, 31, 36, 24, 6, 121, 53, 118, 101, 62, 82,
                   3, 108, 67, 75, 12, 15, 117, 106, 49, 27, 109, 59, 11, 87, 92, 80]

validation_partition = [38, 85, 112, 77, 14, 2, 39, 30, 90, 20, 9, 40]

test_partition = [50, 76, 73, 103, 42, 72, 56, 61, 68, 100, 86, 21, 104, 114, 58, 81,
                  25, 79, 119, 44, 22, 88, 115, 41, 32]

# various portions of training subjects
selected_train_part_25 = random.sample(train_partition, int(len(train_partition) * .25))

selected_train_part_50 = random.sample(train_partition, int(len(train_partition) * .5))

selected_train_part_75 = random.sample(train_partition, int(len(train_partition) * .75))

# Here, we select the portion of training data we are working with.
selected_train_partition = train_partition
train_trials_portion = 1.0  # 1 = 100%
training_trial_idxs = list()  # we start out with random trial data indexes being nothing

# map for all 64 electrodes/channels
full_channel_map: List[str] = ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6', 'FC5', 'FC2', 'FC1',
                               'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6', 'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'P7',
                               'PO2', 'PO1', 'O2', 'O1', 'X', 'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4',
                               'FC3', 'C6', 'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5', 'P6', 'C1',
                               'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2', 'P1', 'CPZ', 'nd', 'Y']

# omits P7, X and Y (EOG), and nd (ground)
standard_1020_channel_map: List[str] = \
    ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6',
     'FC5', 'FC2', 'FC1', 'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6',
     'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'PO2', 'PO1', 'O2', 'O1',
     'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4', 'FC3', 'C6',
     'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5', 'P6',
     'C1', 'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2', 'P1', 'CPZ']

p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']

p300_12_channel_map: List[str] = ['FP1', 'FP2', 'FPZ', 'FZ', 'CZ', 'P3',
                                  'PZ', 'P4', 'PO7', 'PO8', 'POZ', 'OZ']

p300_30_channel_map: List[str] = ['F1', 'FZ', 'F2', 'F4',
                                  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
                                  'C3', 'C1', 'CZ', 'C2', 'C4',
                                  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
                                  'P3', 'P1', 'PZ', 'P2', 'P4',
                                  'PO1', 'POZ', 'PO2',
                                  'O1', 'OZ', 'O2']


def train_val_test_split(p_subjects, group=None, train_part=None):
    train = []
    val = []
    test = []

    if train_part == None:
        train_part = selected_train_partition

    if group is None:
        train = [p_subjects[i] for i in train_part]
        val = [p_subjects[i] for i in validation_partition]
        test = [p_subjects[i] for i in test_partition]
    elif group == 'control':
        train = [p_subjects[i] for i in train_part if not p_subjects[i].alcoholic]
        val = [p_subjects[i] for i in validation_partition if not p_subjects[i].alcoholic]
        test = [p_subjects[i] for i in test_partition if not p_subjects[i].alcoholic]
    elif group == 'alcoholic':
        train = [p_subjects[i] for i in train_part if p_subjects[i].alcoholic]
        val = [p_subjects[i] for i in validation_partition if p_subjects[i].alcoholic]
        test = [p_subjects[i] for i in test_partition if p_subjects[i].alcoholic]

    return train, val, test


def unroll_subject(s):
    N = len(s.trials)
    return [t.data for t in s.trials], \
           [s.alcoholic for i in range(N)], \
           [t.match_cond for t in s.trials], \
           [s.name for i in range(N)]


def clean_all_subjects(subjects):
    dumped = []

    for subject in subjects:
        bad_trials = clean_subject_trials(subject)
        if len(bad_trials) > 0:
            dumped.extend(bad_trials)

    return dumped


def clean_subject_trials(s):
    refined_trials = []
    dumped = []
    for trial in s.trials:
        # rid trials with bad channel data...
        dump_trial = False
        for channel in trial.data:
            count = np.count_nonzero(channel)
            percentage = count / len(channel)
            if percentage < 0.7:
                dump_trial = True
                break

        if not dump_trial:
            refined_trials.append(trial)
        else:
            dumped.append(trial)

    s.trials = refined_trials

    return dumped


def unroll_subjects(subjects, ltype, include_ids=False, portion=1.0):
    data = []
    labels = []
    s_ids = []
    global training_trial_idxs

    for s in subjects:
        dat, alc, match, ids = unroll_subject(s)
        data.extend(dat)
        s_ids.extend(ids)
        if ltype == 'alcoholic':
            labels.extend(alc)
        elif ltype == 'match':
            labels.extend(match)
        else:
            raise ValueError("invalid label type")

    if 1.0 > portion > 0.0:  # Select only subset of trials, if indicated...
        idx_len = int(len(data) * portion)
        len_diff = idx_len - len(training_trial_idxs)
        if len_diff > 0:
            # Now append to the list more indexes that are not already in the list of indexes
            available_idxs = np.setdiff1d(random.sample(range(0, len(data)), len(data)), training_trial_idxs)
            training_trial_idxs.extend(random.sample(list(available_idxs), len_diff))

        data = [data[i] for i in training_trial_idxs]
        labels = [labels[i] for i in training_trial_idxs]
        s_ids = [s_ids[i] for i in training_trial_idxs]

    if include_ids:
        return data, labels, s_ids
    else:
        return data, labels


# keeps only selected EEG channels for all trials under all subjects
def trim_subject_channels(subjects, channel_idxs):
    subject_index = 0
    try:
        for subject in subjects:
            trial_index = 0
            for trial in subject.trials:
                trial.data = np.asarray([trial.data[i] for i in channel_idxs])
                trial_index += 1
            subject_index += 1
    except Exception as ex:
        print("Problem trimming subject channels")
        print("Error: ", ex)


def get_channel_map_indexes(channel_map):
    return [full_channel_map.index(ch) for ch in channel_map]


from functools import partial


def create_base_dataset(data_pickle_file,
                        mask_func=None,
                        ltype='alcoholic',
                        include_ids=False,
                        channel_map=None,
                        group=None,
                        clean_trials=True,
                        train_portion=train_trials_portion,
                        random_seed=23):
    np.random.seed(random_seed)

    with open(data_pickle_file, 'rb') as f:
        p_subjects = pickle.load(f)

    # Remove all bad data: channels with zeros means whole trial is eliminated from the data set
    if clean_trials:
        clean_all_subjects(p_subjects)

    if channel_map is None:
        # default to using 10-20 extended
        channel_map = standard_1020_channel_map

    if channel_map is not None:
        channel_idxs = get_channel_map_indexes(channel_map)
        trim_subject_channels(p_subjects, channel_idxs)

    train, val, test = train_val_test_split(p_subjects, group=group)

    if (include_ids):
        Xtr, ytr, tr_ids = unroll_subjects(train, ltype, True, portion=train_portion)
        Xval, yval, val_ids = unroll_subjects(val, ltype, True)
        Xte, yte, te_ids = unroll_subjects(test, ltype, True)

        return Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids
    else:
        Xtr, ytr = unroll_subjects(train, ltype, portion=train_portion)
        Xval, yval = unroll_subjects(val, ltype)
        Xte, yte = unroll_subjects(test, ltype)

        return Xtr, ytr, Xval, yval, Xte, yte


def create_match_detection_dataset(data_pickle_file=pickle_data_path,
                                   protocol='raw', include_ids=False,
                                   channel_map=None,
                                   group=None,
                                   clean_trials=True,
                                   train_portion=train_trials_portion,
                                   random_seed=23):
    # categories = {'S1 obj', 'S2 match', 'S2 match err', 'S2 nomatch','S2 nomatch err'}

    if protocol == 'raw':

        create_func = partial(create_base_dataset,
                              ltype='match',
                              include_ids=True,
                              channel_map=channel_map,
                              group=group,
                              clean_trials=clean_trials,
                              train_portion=train_portion,
                              random_seed=random_seed)

        Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids = create_func(data_pickle_file)

        keep_ind_tr = [i for i in range(len(ytr)) if ytr[i] in {'S2 match', 'S2 nomatch'}]
        keep_ind_val = [i for i in range(len(yval)) if yval[i] in {'S2 match', 'S2 nomatch'}]
        keep_ind_te = [i for i in range(len(yte)) if yte[i] in {'S2 match', 'S2 nomatch'}]

        Xtr = np.asarray([Xtr[i] for i in keep_ind_tr])
        tr_ids = np.asarray([tr_ids[i] for i in keep_ind_tr])
        ytr = [ytr[i] for i in keep_ind_tr]
        ytr = np.asarray([x == 'S2 match' for x in ytr])

        Xval = np.asarray([Xval[i] for i in keep_ind_val])
        val_ids = np.asarray([val_ids[i] for i in keep_ind_val])
        yval = [yval[i] for i in keep_ind_val]
        yval = np.asarray([x == 'S2 match' for x in yval])

        Xte = np.asarray([Xte[i] for i in keep_ind_te])
        te_ids = np.asarray([te_ids[i] for i in keep_ind_te])
        yte = [yte[i] for i in keep_ind_te]
        yte = np.asarray([x == 'S2 match' for x in yte])

        if (include_ids):
            return Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids
        else:
            return Xtr, ytr, Xval, yval, Xte, yte

    elif protocol == 'diff':

        """"
        This still might need some more debugging due to potentially missing data...
        """

        def diff_it(dataset):

            all_data = []
            all_labels = []
            all_ids = []

            for dat in dataset:
                trials = dat.trials
                mcs = [trials[i].match_cond for i in range(len(trials))]

                data = []
                labels = []
                ids = []
                obj = None
                for i in range(len(mcs)):
                    if mcs[i] == 'S1 obj':
                        obj = trials[i].data

                    if type(obj) != type(None):

                        if mcs[i] == 'S2 match':
                            data.append(trials[i].data - obj)
                            labels.append(1)
                            ids.append(dat.name)
                        if mcs[i] == 'S2 nomatch':
                            data.append(trials[i].data - obj)
                            labels.append(0)
                            ids.append(dat.name)

                all_data.extend(data)
                all_labels.extend(labels)
                all_ids.extend(ids)

            return all_data, all_labels, all_ids

        with open(data_pickle_file, 'rb') as f:
            p_subjects = pickle.load(f)

        if channel_map is not None:
            channel_idxs = get_channel_map_indexes(channel_map)
            trim_subject_channels(p_subjects, channel_idxs)

        train, val, test = train_val_test_split(p_subjects, group=group)

        Xtr, ytr, tr_ids = diff_it(train)
        Xval, yval, val_ids = diff_it(val)
        Xte, yte, te_ids = diff_it(test)

        tr_ids = np.asarray(tr_ids)
        val_ids = np.asarray(val_ids)
        te_ids = np.asarray(te_ids)

        Xtr = np.asarray(Xtr)
        ytr = np.asarray(ytr)
        Xval = np.asarray(Xval)
        yval = np.asarray(yval)
        Xte = np.asarray(Xte)
        yte = np.asarray(yte)

        if (include_ids):
            return Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids
        else:
            return Xtr, ytr, Xval, yval, Xte, yte


def create_alcoholic_detection_dataset(data_pickle_file=pickle_data_path,
                                       include_ids=False,
                                       channel_map=None,
                                       group=None,
                                       clean_trials=True,
                                       train_portion=train_trials_portion,
                                       random_seed=23):
    create_func = partial(create_base_dataset,
                          ltype='alcoholic',
                          include_ids=True,
                          channel_map=channel_map,
                          group=group,
                          clean_trials=clean_trials,
                          train_portion=train_portion,
                          random_seed=random_seed)

    Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids = create_func(data_pickle_file)

    tr_ids = np.asarray(tr_ids)
    val_ids = np.asarray(val_ids)
    te_ids = np.asarray(te_ids)
    Xtr = np.asarray(Xtr)
    ytr = np.asarray(ytr)
    Xval = np.asarray(Xval)
    yval = np.asarray(yval)
    Xte = np.asarray(Xte)
    yte = np.asarray(yte)

    if (include_ids):
        return Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids
    else:
        return Xtr, ytr, Xval, yval, Xte, yte


def test(data_pickle_file=pickle_data_path, channel_map=None):
    with open(data_pickle_file, 'rb') as f:
        p_subjects = pickle.load(f)

    if channel_map is not None:
        channel_idxs = get_channel_map_indexes(channel_map)
        trim_subject_channels(p_subjects, channel_idxs)

    Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids = \
        create_match_detection_dataset(include_ids=True)

    Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids = \
        create_match_detection_dataset(protocol='diff', include_ids=True)

    Xtr, ytr, tr_ids, Xval, yval, val_ids, Xte, yte, te_ids = \
        create_alcoholic_detection_dataset(include_ids=True)

    embed()


if __name__ == '__main__':
    test()
