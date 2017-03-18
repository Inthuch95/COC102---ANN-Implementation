'''
Created on Mar 18, 2017

@author: Inthuch Therdchanakul    
'''
import pickle

with open("../Results/standard_5_e20000/standard_5_e20000.pickle", "rb") as f:
    clf = pickle.load(f)

print("rmse: ", min(clf.rmse))
print("msre: ", min(clf.msre))