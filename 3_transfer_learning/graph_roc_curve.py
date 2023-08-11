from typing import List
import numpy as np
from keras import models
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib import pyplot as plt
import pickle

predictions = []

with open('validation_data.pkl', 'rb') as file:
        validation_data = pickle.load(file)

with open('validation_labels.pkl', 'rb') as file:
        validation_labels = pickle.load(file)

# Recalculate Predictions?
network = models.load_model('new.h5')

# get predictions for validation data
for i in validation_data:
    pred = network.predict(np.array([i,]), verbose=0)
    pred = pred[0][0]
    predictions.append(pred)

# pickle the data
with open('pred.pkl', 'wb') as file:
    pickle.dump(predictions, file)

# calculate roc curve
fpr , tpr , thresholds = roc_curve (validation_labels, predictions)

# calculate area under roc curve
auc_score = roc_auc_score(validation_labels, predictions)
print(f"Area under curve: {auc_score}")

# plot curve
def plot_roc_curve(fpr,tpr): 
  plt.title(f"Area under curve: {auc_score}")
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    

plot_roc_curve(fpr,tpr)