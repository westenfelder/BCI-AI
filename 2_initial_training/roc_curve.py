from typing import List
import numpy as np
from keras import models
import dataset as api
import random
import time
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib import pyplot as plt
import pickle

# Seed random number generator
random.seed(time.time())
RANDOM_SEED = random.randint(0, 100)
RECALC = False

# Define Channel Map
p300_8_channel_map: List[str] = ['FZ', 'CZ', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'OZ']
predictions = []

# Load Validation Data
data = api.create_match_detection_dataset(data_pickle_file='data.dat', channel_map=p300_8_channel_map, random_seed=RANDOM_SEED)
validation_data = data[2]
validation_labels = data[3]

# Recalculate Predictions?
if RECALC:
    network = models.load_model('additional_data.h5')

    # get predictions for validation data
    for i in validation_data:
        pred = network.predict(np.array([i,]), verbose=0)
        pred = pred[0][0]
        predictions.append(pred)

    # pickle the data
    with open('pred.pkl', 'wb') as file:
        pickle.dump(predictions, file)

# Otherwise load pickled data
else:
    with open('pred.pkl', 'rb') as file:
        predictions = pickle.load(file)

# calculate roc curve
fpr , tpr , thresholds = roc_curve (validation_labels, predictions)

# calculate area under roc curve
auc_score = roc_auc_score(validation_labels, predictions)
print(f"Area under curve: {auc_score}")

# plot curve
def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    

plot_roc_curve(fpr,tpr)