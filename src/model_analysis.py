'''
Created on Mar 18, 2017

@author: Inthuch Therdchanakul    
'''
import pickle
import matplotlib.pyplot as plt

training_type = "momentum"
n_hidden_units = "3"
epoch = "2000"

path = "Simulations/" + training_type + "_" + n_hidden_units + "_e" + epoch + "/"
predictions_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_PREDICTIONS.pickle"
rmse_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_RMSE.pickle"
clf_filename = path + training_type + "_" + n_hidden_units + "_e" + epoch + "_MODEL.pickle"
with open(predictions_filename, "rb") as f:
    predictions = pickle.load(f)
with open(rmse_filename, "rb") as f:
    rmse = pickle.load(f)  
with open(clf_filename, "rb") as f:
    clf = pickle.load(f)   

print(clf.best_network)
predictions.set_ylim(-100, 900)
plt.show()