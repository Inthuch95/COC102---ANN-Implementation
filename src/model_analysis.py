'''
Created on Mar 18, 2017

@author: Inthuch Therdchanakul    
'''
import pickle

with open("../Results/m_5_e20000/m_5_e20000.pickle", "rb") as f:
    clf = pickle.load(f)
    
msre_val, msre_idx = min((msre_val, msre_idx) for (msre_idx, msre_val) in enumerate(clf.msre))
rmse_val, rmse_idx = min((rmse_val, rmse_idx) for (rmse_idx, rmse_val) in enumerate(clf.rmse))
print("index: %s, msre: %s" % (str(msre_idx), str(msre_val)))
print("index: %s, rmse: %s" % (str(rmse_idx), str(rmse_val)))
